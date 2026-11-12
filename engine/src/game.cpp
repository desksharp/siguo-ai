// 对局状态机：走法生成、落子与碰子、出局与终局、按观察者的硬约束信念追踪
#include <algorithm>

#include "junqi/junqi.h"

namespace junqi {
namespace {
template <class Arr>
uint32_t NextGen(uint32_t& gen, Arr& stamps) {
  if (++gen == 0) {
    for (auto& s : stamps) s = {};
    gen = 1;
  }
  return gen;
}
}  // namespace

Game::Game(const RuleConfig& cfg) : cfg_(cfg), B_(&Board::Get(cfg)) {
  Reset(0);
}

bool Game::visible(int observer, int owner) const {
  if (observer == owner || cfg_.visibility == kVisAll) return true;
  return cfg_.visibility == kVisTeam && cfg_.mode == kModeFour && team_[observer] == team_[owner];
}

int Game::pieces_alive(int seat) const {
  int n = 0;
  for (int k = 0; k < kPiecesPerSeat; ++k) n += pieces_[seat * kPiecesPerSeat + k].alive;
  return n;
}

void Game::Reset(uint64_t seed, const int8_t* setups, const int* setup_kinds) {
  seed_ = seed;
  Rng rng(seed);
  const bool duel = cfg_.mode == kModeDuel;
  for (int s = 0; s < kSeats; ++s) {
    active_[s] = duel ? (s == 0 || s == 2) : 1;
    team_[s] = duel ? (s == 0 ? 0 : (s == 2 ? 1 : 2)) : int8_t(s % 2);
    alive_[s] = active_[s];
    flag_revealed_[s] = 0;
  }
  setups_.fill(-1);
  for (int s = 0; s < kSeats; ++s) {
    if (!active_[s]) continue;
    int8_t* camp = &setups_[s * kCampNodes];
    if (setups != nullptr && ValidSetup(setups + s * kCampNodes)) {
      std::copy(setups + s * kCampNodes, setups + (s + 1) * kCampNodes, camp);
    } else {
      GenerateSetup(setup_kinds ? setup_kinds[s] : kSetupHeuristic, rng, camp);
    }
  }
  board_.fill(-1);
  for (auto& p : pieces_) p = Piece{};
  for (int s = 0; s < kSeats; ++s) {
    for (int k = 0; k < kPiecesPerSeat; ++k) pieces_[s * kPiecesPerSeat + k].owner = int8_t(s);
    if (!active_[s]) continue;
    int k = 0;
    for (int pos = 0; pos < kCampNodes; ++pos) {
      const int t = setups_[s * kCampNodes + pos];
      if (t < 0) continue;
      const int pid = s * kPiecesPerSeat + k++;
      const int node = CampNode(s, pos / 5, pos % 5);
      Piece& p = pieces_[pid];
      p.type = int8_t(t);
      p.node = p.start_node = int16_t(node);
      p.alive = 1;
      board_[node] = int16_t(pid);
    }
  }
  cur_ = 0;
  ply_ = plies_since_combat_ = 0;
  winner_ = -1;
  end_reason_ = kEndNone;
  over_ = false;
  history_.clear();
  outcomes_.clear();
  roster_complete_ = true;
  InitBelief(true);
}

bool Game::LoadPosition(const int8_t* types, const int8_t* owners, int current_seat, uint64_t seed) {
  Reset(seed);
  board_.fill(-1);
  for (auto& p : pieces_) p = Piece{};
  int count[kSeats] = {};
  for (int s = 0; s < kSeats; ++s)
    for (int k = 0; k < kPiecesPerSeat; ++k) pieces_[s * kPiecesPerSeat + k].owner = int8_t(s);
  for (int n = 0; n < kNodes; ++n) {
    const int t = types[n], o = owners[n];
    if (t < 0) continue;
    if (t >= kTypes || o < 0 || o >= kSeats || !active_[o] || count[o] >= kPiecesPerSeat) return false;
    if (B_->is_base[n] == 0 && t == kFlag) return false;
    const int pid = o * kPiecesPerSeat + count[o]++;
    Piece& p = pieces_[pid];
    p.type = int8_t(t);
    p.node = p.start_node = int16_t(n);
    p.alive = 1;
    p.moved = B_->seat_of[n] != o;  // 不在本方营区的子一定动过
    board_[n] = int16_t(pid);
  }
  for (int s = 0; s < kSeats; ++s) alive_[s] = active_[s] && count[s] > 0;
  if (current_seat < 0 || current_seat >= kSeats || !alive_[current_seat]) return false;
  setups_.fill(-1);
  cur_ = current_seat;
  roster_complete_ = false;
  InitBelief(false);
  return true;
}

template <class Probe>
void Game::Slide(int from, bool engineer, Probe&& probe) const {
  if (engineer) {  // 工兵：铁路上任意拐弯，经过的点必须为空
    const uint32_t vs = NextGen(visit_gen_, visit_stamp_);
    int q[kNodes];
    int qh = 0, qt = 0;
    q[qt++] = from;
    visit_stamp_[from] = vs;
    while (qh < qt) {
      const int a = q[qh++];
      for (int16_t b : B_->rail_adj[a]) {
        if (visit_stamp_[b] == vs) continue;
        visit_stamp_[b] = vs;
        if (probe(b)) q[qt++] = b;
      }
    }
    return;
  }
  // 其它棋子：沿“直行连续表”滑行，弧线按 arc_mode 视为直线
  const uint32_t es = NextGen(edge_gen_, edge_stamp_);
  int16_t sa[256], sb[256];
  int sp = 0;
  for (int16_t b : B_->rail_adj[from]) {
    sa[sp] = int16_t(from);
    sb[sp] = b;
    ++sp;
  }
  while (sp > 0) {
    --sp;
    const int a = sa[sp], b = sb[sp];
    const int ia = B_->rail_index[b][a];
    if (edge_stamp_[b][ia] == es) continue;
    edge_stamp_[b][ia] = es;
    if (b == from || !probe(b)) continue;
    const uint8_t m = B_->straight[b][ia];
    const auto& ra = B_->rail_adj[b];
    for (size_t j = 0; j < ra.size(); ++j) {
      if (((m >> j) & 1u) && sp < 256) {
        sa[sp] = int16_t(b);
        sb[sp] = ra[j];
        ++sp;
      }
    }
  }
}

void Game::AddPieceMoves(int pid, bool mobile, bool engineer, Move* out, int& n, int cap) const {
  const Piece& p = pieces_[pid];
  if (!p.alive || !mobile) return;
  const int from = p.node;
  if (B_->is_base[from]) return;  // 进入大本营的子不能再动
  const uint32_t ds = NextGen(dest_gen_, dest_stamp_);
  const int my_team = team_[p.owner];
  auto emit = [&](int d) {
    if (dest_stamp_[d] == ds) return;
    dest_stamp_[d] = ds;
    if (n < cap) out[n] = Move{int16_t(from), int16_t(d)};
    ++n;
  };
  auto probe = [&](int d) -> bool {
    const int occ = board_[d];
    if (occ < 0) {
      emit(d);
      return true;
    }
    if (team_[pieces_[occ].owner] != my_team && !B_->is_camp[d]) emit(d);  // 行营内的子不可被攻击
    return false;
  };
  for (int16_t d : B_->adj[from]) probe(d);
  if (B_->is_rail[from]) Slide(from, engineer, probe);
}

int Game::GenMoves(int seat, Move* out, int cap) const {
  int n = 0;
  if (seat < 0 || seat >= kSeats || !active_[seat] || !alive_[seat]) return 0;
  for (int k = 0; k < kPiecesPerSeat; ++k) {
    const Piece& p = pieces_[seat * kPiecesPerSeat + k];
    AddPieceMoves(seat * kPiecesPerSeat + k, p.type != kFlag && p.type != kMine, p.type == kEngineer, out, n, cap);
  }
  return n;
}

int Game::GenMovesBelief(int seat, int observer, Move* out, int cap) const {
  int n = 0;
  if (seat < 0 || seat >= kSeats || !active_[seat] || !alive_[seat]) return 0;
  constexpr Mask kMobile = Mask(kAllTypes & ~(Bit(kFlag) | Bit(kMine)));
  for (int k = 0; k < kPiecesPerSeat; ++k) {
    const int pid = seat * kPiecesPerSeat + k;
    const Mask m = belief_[observer][pid];
    AddPieceMoves(pid, (m & kMobile) != 0, (m & Bit(kEngineer)) != 0, out, n, cap);
  }
  return n;
}

bool Game::IsLegal(int from, int to) const {
  if (over_ || from < 0 || from >= kNodes || to < 0 || to >= kNodes) return false;
  const int pid = board_[from];
  if (pid < 0 || pieces_[pid].owner != cur_) return false;
  Move buf[kNodes];
  int n = 0;
  const int t = pieces_[pid].type;
  AddPieceMoves(pid, t != kFlag && t != kMine, t == kEngineer, buf, n, kNodes);
  for (int i = 0; i < n && i < kNodes; ++i)
    if (buf[i].to == to) return true;
  return false;
}

bool Game::StraightReach(int pid, int to) const {
  const Piece& p = pieces_[pid];
  if (p.node < 0) return false;
  if (B_->edge_kind[p.node][to] != 0) return true;
  if (!B_->is_rail[p.node]) return false;
  bool hit = false;
  Slide(p.node, false, [&](int d) -> bool {
    if (d == to) hit = true;
    return board_[d] < 0;
  });
  return hit;
}

int Game::Step(int from, int to) {
  if (!IsLegal(from, to)) return kOutIllegal;
  const int pid = board_[from];
  Piece& p = pieces_[pid];
  // 公开信息：不是一步、也不是直行可达 → 只能是工兵拐弯
  const bool engineer_reveal = p.type == kEngineer && !StraightReach(pid, to);
  const int did = board_[to];
  int outcome = kOutMove;
  int dead[2] = {-1, -1};
  board_[from] = -1;
  if (did < 0) {
    board_[to] = int16_t(pid);
    p.node = int16_t(to);
  } else {
    Piece& d = pieces_[did];
    outcome = Resolve(p.type, d.type);
    switch (outcome) {
      case kOutKill:
      case kOutCaptureFlag:
        d.alive = 0;
        d.node = -1;
        board_[to] = int16_t(pid);
        p.node = int16_t(to);
        ++p.combats;
        dead[0] = did;
        break;
      case kOutKilled:
        p.alive = 0;
        p.node = -1;
        ++d.combats;
        dead[0] = pid;
        break;
      default:  // kOutBoth
        p.alive = 0;
        p.node = -1;
        d.alive = 0;
        d.node = -1;
        board_[to] = -1;
        dead[0] = pid;
        dead[1] = did;
        break;
    }
  }
  p.moved = 1;
  p.last_moved_ply = ply_;
  plies_since_combat_ = did >= 0 ? 0 : plies_since_combat_ + 1;
  history_.push_back(Move{int16_t(from), int16_t(to)});
  outcomes_.push_back(int8_t(outcome));

  PublicMoved(pid);
  if (engineer_reveal) PublicSetMask(pid, Bit(kEngineer));
  if (did >= 0) {
    if (outcome == kOutCaptureFlag) PublicSetMask(did, Bit(kFlag));
    if (cfg_.reveal_flag_on_marshal_death) {
      for (int x : dead) {
        if (x < 0 || pieces_[x].type != kMarshal) continue;
        const int owner = pieces_[x].owner;
        flag_revealed_[owner] = 1;  // 司令阵亡：亮军旗
        PublicSetMask(x, Bit(kMarshal));
        for (int k = 0; k < kPiecesPerSeat; ++k) {
          const int f = owner * kPiecesPerSeat + k;
          if (pieces_[f].type == kFlag) PublicSetMask(f, Bit(kFlag));
        }
      }
    }
    PublicCombat(pid, did, outcome);
  }
  ++ply_;
  if (outcome == kOutCaptureFlag) Eliminate(pieces_[did].owner);
  Advance();
  return outcome;
}

void Game::Eliminate(int seat) {
  alive_[seat] = 0;
  if (!cfg_.remove_eliminated) return;
  for (int k = 0; k < kPiecesPerSeat; ++k) {
    Piece& p = pieces_[seat * kPiecesPerSeat + k];
    if (!p.alive) continue;
    board_[p.node] = -1;
    p.alive = 0;
    p.node = -1;
  }
}

void Game::Advance() {
  for (;;) {
    bool team_alive[2] = {false, false};
    for (int s = 0; s < kSeats; ++s)
      if (active_[s] && alive_[s] && team_[s] < 2) team_alive[team_[s]] = true;
    if (!team_alive[0] || !team_alive[1]) {
      over_ = true;
      end_reason_ = kEndTeamEliminated;
      winner_ = team_alive[0] ? 0 : (team_alive[1] ? 1 : -1);
      return;
    }
    if (ply_ >= cfg_.max_plies) {
      over_ = true;
      end_reason_ = kEndMaxPlies;
      return;
    }
    if (plies_since_combat_ >= cfg_.max_plies_without_combat) {
      over_ = true;
      end_reason_ = kEndNoCombat;
      return;
    }
    int nxt = cur_;
    do {
      nxt = (nxt + 1) % kSeats;
    } while (!(active_[nxt] && alive_[nxt]));
    cur_ = nxt;
    if (GenMoves(cur_, nullptr, 0) > 0) return;
    Eliminate(cur_);  // 无子可动：出局
  }
}

// ------------------------------------------------------------------ belief
void Game::InitBelief(bool from_setup) {
  for (int o = 0; o < kSeats; ++o) {
    for (int pid = 0; pid < kPieces; ++pid) {
      const Piece& p = pieces_[pid];
      Mask m = 0;
      if (p.type >= 0) {
        if (visible(o, p.owner)) {
          m = Bit(p.type);
        } else {  // 布阵约束：军旗只在大本营、地雷只在后两排、炸弹不在第一排
          const int r = B_->row[p.start_node], c = B_->col[p.start_node];
          const bool home = B_->seat_of[p.start_node] == p.owner;
          m = kAllTypes;
          if (!home || !IsBaseCell(r, c)) m &= Mask(~Bit(kFlag));
          if (!home || r < 4) m &= Mask(~Bit(kMine));
          if (from_setup && r == 0) m &= Mask(~Bit(kBomb));
        }
      }
      belief_[o][pid] = m;
    }
  }
}

void Game::PublicMoved(int pid) { PublicSetMask(pid, Mask(kAllTypes & ~(Bit(kFlag) | Bit(kMine)))); }

void Game::PublicSetMask(int pid, Mask keep) {
  const int owner = pieces_[pid].owner;
  for (int o = 0; o < kSeats; ++o) {
    const Mask before = belief_[o][pid];
    const Mask after = before & keep;
    if (after == before) continue;
    belief_[o][pid] = after;
    Propagate(o, owner);
  }
}

void Game::PublicCombat(int a, int d, int outcome) {
  for (int o = 0; o < kSeats; ++o) {
    const Mask ma = belief_[o][a], md = belief_[o][d];
    Mask na = 0, nd = 0;
    for (int t = 0; t < kTypes; ++t) {
      if ((ma & Bit(t)) && (CompatDefender(outcome, t) & md)) na |= Bit(t);
      if ((md & Bit(t)) && (CompatAttacker(outcome, t) & ma)) nd |= Bit(t);
    }
    if (na != ma) {
      belief_[o][a] = na;
      Propagate(o, pieces_[a].owner);
    }
    if (nd != md) {
      belief_[o][d] = nd;
      Propagate(o, pieces_[d].owner);
    }
  }
}

// 数量约束传播：已确定数量用满 → 其余子排除；候选数量正好等于配额 → 候选全部确定
void Game::Propagate(int o, int owner) {
  Mask* m = &belief_[o][owner * kPiecesPerSeat];
  const Piece* ps = &pieces_[owner * kPiecesPerSeat];
  for (int guard = 0; guard < 16; ++guard) {
    bool changed = false;
    for (int t = 0; t < kTypes; ++t) {
      const Mask b = Bit(t);
      int fixed = 0, cand = 0;
      for (int k = 0; k < kPiecesPerSeat; ++k) {
        if (ps[k].type < 0) continue;
        fixed += m[k] == b;
        cand += (m[k] & b) != 0;
      }
      if (cand == fixed) continue;
      if (fixed == kQuota[t]) {
        for (int k = 0; k < kPiecesPerSeat; ++k)
          if (ps[k].type >= 0 && m[k] != b && (m[k] & b)) {
            m[k] &= Mask(~b);
            changed = true;
          }
      } else if (roster_complete_ && cand == kQuota[t]) {
        for (int k = 0; k < kPiecesPerSeat; ++k)
          if (ps[k].type >= 0 && m[k] != b && (m[k] & b)) {
            m[k] = b;
            changed = true;
          }
      }
    }
    if (!changed) break;
  }
}

}  // namespace junqi
