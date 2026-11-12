// 内置智能体：随机 AI 与规则 AI（硬约束信念 + 期望子力收益 + 威胁/推进启发式）
#include <algorithm>
#include <cmath>

#include "junqi/junqi.h"

namespace junqi {
namespace {

constexpr int kMoveBuf = 1024;
// 子力价值：军旗 地雷 炸弹 工兵 排长 连长 营长 团长 旅长 师长 军长 司令
const float kValue[kTypes] = {1000.0f, 1.0f, 4.5f, 2.0f, 1.0f, 1.5f, 2.2f, 3.0f, 4.0f, 5.5f, 7.5f, 10.0f};

struct Dist {
  float p[kTypes];
};

// 观察者对某子兵种的分布：硬约束掩码 × 按配额的先验。
// mine_boost > 1：对方后两排里还可能是地雷的子，地雷先验放大（动过的子掩码里已经没有地雷）
Dist BeliefDist(const Game& g, int observer, int pid, float mine_boost = 1.0f) {
  Dist d{};
  const Mask m = g.belief(observer, pid);
  float tot = 0.0f;
  for (int t = 0; t < kTypes; ++t)
    if (m & Bit(t)) tot += (d.p[t] = float(kQuota[t]));
  const Piece& pc = g.piece(pid);
  if (mine_boost != 1.0f && (m & Bit(kMine)) && g.board().seat_of[pc.node] == pc.owner && g.board().row[pc.node] >= 4) {
    tot += d.p[kMine] * (mine_boost - 1.0f);
    d.p[kMine] *= mine_boost;
  }
  if (tot <= 0.0f) {
    for (int t = 0; t < kTypes; ++t) d.p[t] = 1.0f / kTypes;
    return d;
  }
  for (float& v : d.p) v /= tot;
  return d;
}

float AttackGain(int ta, const Dist& d, float* survive) {
  float gain = 0.0f, s = 0.0f;
  for (int t = 0; t < kTypes; ++t) {
    if (d.p[t] <= 0.0f) continue;
    switch (Resolve(ta, t)) {
      case kOutCaptureFlag:
        gain += d.p[t] * kValue[kFlag];
        s += d.p[t];
        break;
      case kOutKill:
        gain += d.p[t] * kValue[t];
        s += d.p[t];
        break;
      case kOutKilled:
        gain -= d.p[t] * kValue[ta];
        break;
      case kOutBoth:
        gain += d.p[t] * (kValue[t] - kValue[ta]);
        break;
      default:
        break;
    }
  }
  if (survive) *survive = s;
  return gain;
}

// 敌子（分布 e）主动攻击我方 ta 时我方的期望损失比例
float BeatProb(const Dist& e, int ta) {
  float p = 0.0f;
  for (int t = kBomb; t < kTypes; ++t) {  // 只有可动的子能来攻击
    if (e.p[t] <= 0.0f) continue;
    const int o = Resolve(t, ta);
    if (o == kOutKill || o == kOutCaptureFlag) p += e.p[t];
    if (o == kOutBoth) p += 0.5f * e.p[t];
  }
  return p;
}

}  // namespace

Move RandomAct(const Game& g, Rng& rng) {
  Move buf[kMoveBuf];
  const int n = std::min(g.LegalMoves(buf, kMoveBuf), kMoveBuf);
  if (n <= 0) return Move{-1, -1};
  return buf[rng.Below(n)];
}

Move RuleAct(const Game& g, Rng& rng, float temperature, float mine_boost) {
  const Board& B = g.board();
  const int me = g.current_seat();
  const int my_team = g.team_of(me);
  Move moves[kMoveBuf];
  const int n = std::min(g.LegalMoves(moves, kMoveBuf), kMoveBuf);
  if (n <= 0) return Move{-1, -1};

  // 敌方威胁：每个节点可能被哪些敌子打到（按我方信念推算，规则 AI 不偷看）
  constexpr int kMaxAtk = 12;
  int16_t atk[kNodes][kMaxAtk];
  uint8_t natk[kNodes] = {};
  Dist enemy_dist[kPieces];
  bool have_dist[kPieces] = {};
  Move ebuf[kMoveBuf];
  for (int s = 0; s < kSeats; ++s) {
    if (!g.active(s) || !g.seat_alive(s) || g.team_of(s) == my_team) continue;
    const int en = std::min(g.GenMovesBelief(s, me, ebuf, kMoveBuf), kMoveBuf);
    for (int i = 0; i < en; ++i) {
      const int to = ebuf[i].to, pid = g.piece_at(ebuf[i].from);
      if (natk[to] < kMaxAtk) atk[to][natk[to]++] = int16_t(pid);
      if (!have_dist[pid]) {
        enemy_dist[pid] = BeliefDist(g, me, pid, mine_boost);
        have_dist[pid] = true;
      }
    }
  }
  auto risk = [&](int node, int ta, int ignore_pid) {
    if (B.is_camp[node]) return 0.0f;
    float r = 0.0f;
    for (int k = 0; k < natk[node]; ++k) {
      const int e = atk[node][k];
      if (e == ignore_pid) continue;
      r = std::max(r, BeatProb(enemy_dist[e], ta));
    }
    return r * kValue[ta];
  };

  // 我方军旗与对它的威胁
  int my_flag_node = -1;
  for (int k = 0; k < kPiecesPerSeat; ++k) {
    const Piece& p = g.piece(me * kPiecesPerSeat + k);
    if (p.alive && p.type == kFlag) my_flag_node = p.node;
  }
  bool flag_threat[kPieces] = {};
  if (my_flag_node >= 0)
    for (int k = 0; k < natk[my_flag_node]; ++k) flag_threat[atk[my_flag_node][k]] = true;

  // 推进目标：已亮的敌旗，否则敌方两个大本营
  int targets[8];
  int nt = 0;
  for (int s = 0; s < kSeats; ++s) {
    if (!g.active(s) || !g.seat_alive(s) || g.team_of(s) == my_team) continue;
    bool found = false;
    if (g.flag_revealed(s))
      for (int k = 0; k < kPiecesPerSeat; ++k) {
        const Piece& p = g.piece(s * kPiecesPerSeat + k);
        if (p.alive && p.type == kFlag) {
          targets[nt++] = p.node;
          found = true;
        }
      }
    if (!found) {
      targets[nt++] = CampNode(s, 5, 1);
      targets[nt++] = CampNode(s, 5, 3);
    }
  }
  auto target_dist = [&](int node) {
    int d = 255;
    for (int i = 0; i < nt; ++i) d = std::min<int>(d, B.dist[node][targets[i]]);
    return float(d);
  };

  float score[kMoveBuf];
  float best = -1e30f;
  for (int i = 0; i < n; ++i) {
    const int from = moves[i].from, to = moves[i].to;
    const int pid = g.piece_at(from);
    const int ta = g.piece(pid).type;
    const int def = g.piece_at(to);
    float s = 0.0f, survive = 1.0f;
    if (def >= 0) {
      const Dist dd = BeliefDist(g, me, def, mine_boost);
      s += AttackGain(ta, dd, &survive);
      const Piece& dp = g.piece(def);
      if (ta == kEngineer) s += 1.5f * dd.p[kMine] * (B.row[dp.node] >= 4 ? 1.5f : 1.0f);
      if (flag_threat[def]) s += 20.0f * (1.0f - (1.0f - survive) * 0.5f);
    }
    const float risk_from = risk(from, ta, -1);
    const float risk_to = risk(to, ta, def);
    s += 0.6f * (risk_from - survive * risk_to);
    const float adv = target_dist(from) - target_dist(to);
    s += (ta >= kBattalion ? 0.12f : 0.07f) * adv * survive;
    if (B.is_base[to] && def < 0) s -= 3.0f;  // 进大本营会被困死
    if (B.is_camp[to] && def < 0) s += 0.1f;
    score[i] = s;
    best = std::max(best, s);
  }
  if (temperature <= 1e-4f) {
    int bi = 0;
    for (int i = 1; i < n; ++i)
      if (score[i] > score[bi] || (score[i] == score[bi] && rng.Below(2) == 0)) bi = i;
    return moves[bi];
  }
  double total = 0.0;
  double w[kMoveBuf];
  for (int i = 0; i < n; ++i) total += (w[i] = std::exp((score[i] - best) / temperature));
  double u = rng.Uniform() * total;
  for (int i = 0; i < n; ++i) {
    if (u < w[i]) return moves[i];
    u -= w[i];
  }
  return moves[n - 1];
}

Move AgentAct(int kind, float param, const Game& g, Rng& rng) {
  switch (kind) {
    case kAgentRule:
      return RuleAct(g, rng, param);
    case kAgentExploit:
      return RuleAct(g, rng, param, 4.0f);
    default:
      return RandomAct(g, rng);
  }
}

}  // namespace junqi
