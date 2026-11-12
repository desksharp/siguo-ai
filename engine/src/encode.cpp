// 观测编码：以观察者为 0 号座位旋转，输出节点特征、全局特征与合法着掩码
#include <cmath>
#include <cstring>

#include "junqi/junqi.h"

namespace junqi {
namespace {
constexpr int kReachBuf = 1024;

// 某座位所有子下一手可能到达（含攻击）的节点：按观察者的信念推算，不泄漏隐藏兵种
void MarkReach(const Game& g, int seat, int observer, uint8_t* mark) {
  static thread_local Move buf[kReachBuf];
  const int n = g.GenMovesBelief(seat, observer, buf, kReachBuf);
  for (int i = 0; i < n && i < kReachBuf; ++i) mark[buf[i].to] = 1;
}
}  // namespace

void EncodeObservation(const Game& g, int o, float* nf, float* gf, uint8_t* legal) {
  const Board& B = g.board();
  const RuleConfig& cfg = g.cfg();
  std::memset(nf, 0, sizeof(float) * kNodes * kNodeFeatures);
  std::memset(gf, 0, sizeof(float) * kGlobalFeatures);

  uint8_t reach_any[kNodes] = {}, reach_next[kNodes] = {}, reach_partner[kNodes] = {};
  int next_enemy = -1;
  for (int k = 1; k < kSeats; ++k) {
    const int s = (o + k) % kSeats;
    if (g.active(s) && g.seat_alive(s) && g.team_of(s) != g.team_of(o)) {
      next_enemy = s;
      break;
    }
  }
  for (int s = 0; s < kSeats; ++s) {
    if (s == o || !g.active(s) || !g.seat_alive(s)) continue;
    if (g.team_of(s) == g.team_of(o)) {
      MarkReach(g, s, o, reach_partner);
    } else {
      MarkReach(g, s, o, s == next_enemy ? reach_next : reach_any);
    }
  }
  for (int n = 0; n < kNodes; ++n) reach_any[n] |= reach_next[n];

  const auto& hist = g.history();
  const int ply = g.ply();
  for (int n = 0; n < kNodes; ++n) {
    float* f = nf + B.rel[o][n] * kNodeFeatures;
    f[0] = B.is_rail[n];
    f[1] = B.is_camp[n];
    f[2] = B.is_base[n];
    f[3 + (B.seat_of[n] < 0 ? 4 : (B.seat_of[n] - o + kSeats) % kSeats)] = 1.0f;
    const int pid = g.piece_at(n);
    if (pid < 0) {
      f[8] = 1.0f;
    } else {
      const Piece& p = g.piece(pid);
      const int rs = (p.owner - o + kSeats) % kSeats;
      f[9 + rs] = 1.0f;
      const bool friendly = g.team_of(p.owner) == g.team_of(o);
      f[13] = friendly;
      f[14] = !friendly;
      const Mask m = g.belief(o, pid);
      for (int t = 0; t < kTypes; ++t) f[15 + t] = (m >> t) & 1u;
      const int pc = Popcount(m);
      f[27] = pc / float(kTypes);
      f[28] = p.moved;
      f[29] = p.moved ? std::exp(-(ply - p.last_moved_ply) / 16.0f) : 0.0f;
      f[30] = p.combats >= 3 ? 1.0f : p.combats / 3.0f;
      f[31] = pc == 1;
    }
    f[38] = reach_any[n];
    f[39] = reach_next[n];
    f[40] = reach_partner[n];
  }
  for (int k = 0; k < 3 && k < int(hist.size()); ++k) {
    const Move& mv = hist[hist.size() - 1 - k];
    nf[B.rel[o][mv.from] * kNodeFeatures + 32 + 2 * k] = 1.0f;
    nf[B.rel[o][mv.to] * kNodeFeatures + 33 + 2 * k] = 1.0f;
  }

  gf[0] = float(ply) / float(cfg.max_plies);
  gf[1] = float(g.plies_since_combat()) / float(cfg.max_plies_without_combat);
  for (int s = 0; s < kSeats; ++s) {
    const int rs = (s - o + kSeats) % kSeats;
    gf[2 + rs] = g.active(s) && g.seat_alive(s);
    gf[6 + rs] = g.pieces_alive(s) / float(kPiecesPerSeat);
    gf[10 + rs] = g.flag_revealed(s);
  }
  gf[14] = cfg.mode == kModeDuel;
  gf[15] = cfg.visibility == kVisTeam;

  static thread_local Move buf[kReachBuf];
  const int nm = g.GenMoves(o, buf, kReachBuf);
  gf[16] = nm / 100.0f;
  if (legal != nullptr) {
    std::memset(legal, 0, size_t(kNodes) * kNodes);
    for (int i = 0; i < nm && i < kReachBuf; ++i)
      legal[B.rel[o][buf[i].from] * kNodes + B.rel[o][buf[i].to]] = 1;
  }
}

void EncodeTrueTypes(const Game& g, int o, int8_t* out) {
  const Board& B = g.board();
  for (int n = 0; n < kNodes; ++n) {
    const int pid = g.piece_at(n);
    out[B.rel[o][n]] = pid < 0 ? int8_t(-1) : g.piece(pid).type;
  }
}

}  // namespace junqi
