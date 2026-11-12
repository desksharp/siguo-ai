// C ABI：供 Python ctypes 调用（不依赖 Python 头文件与 pybind11，Linux/Windows 通用）
#include <cstdio>
#include <cstring>
#include <vector>

#include "junqi/junqi.h"

#if defined(_WIN32)
#define JQ_API extern "C" __declspec(dllexport)
#else
#define JQ_API extern "C" __attribute__((visibility("default")))
#endif

using namespace junqi;

namespace {
Game* G(void* h) { return static_cast<Game*>(h); }
}  // namespace

JQ_API int jq_version() { return 1; }

JQ_API void jq_dims(int32_t* out) {
  out[0] = kNodes;
  out[1] = kNodeFeatures;
  out[2] = kGlobalFeatures;
  out[3] = kTypes;
  out[4] = kSeats;
  out[5] = kCampNodes;
}

JQ_API void jq_board_info(int32_t arc_mode, int32_t center_middle_rail, int32_t corner, int16_t* xy, uint8_t* flags, int8_t* srcs,
                          int16_t* rel, uint8_t* edge_kind) {
  const Board& B = Board::Get(arc_mode, center_middle_rail != 0, corner != 0);
  for (int n = 0; n < kNodes; ++n) {
    if (xy) {
      xy[2 * n] = B.x[n];
      xy[2 * n + 1] = B.y[n];
    }
    if (flags) {
      flags[3 * n] = B.is_rail[n];
      flags[3 * n + 1] = B.is_camp[n];
      flags[3 * n + 2] = B.is_base[n];
    }
    if (srcs) {
      srcs[3 * n] = B.seat_of[n];
      srcs[3 * n + 1] = B.row[n];
      srcs[3 * n + 2] = B.col[n];
    }
    if (edge_kind)
      for (int m = 0; m < kNodes; ++m) edge_kind[n * kNodes + m] = B.edge_kind[n][m];
  }
  if (rel)
    for (int o = 0; o < kSeats; ++o)
      for (int n = 0; n < kNodes; ++n) rel[o * kNodes + n] = B.rel[o][n];
}

// a→b→c 是否为非工兵可连续直行的铁路
JQ_API int jq_board_straight(int32_t arc_mode, int32_t center_middle_rail, int32_t corner, int a, int b, int c) {
  const Board& B = Board::Get(arc_mode, center_middle_rail != 0, corner != 0);
  if (a < 0 || b < 0 || c < 0 || a >= kNodes || b >= kNodes || c >= kNodes) return 0;
  const int ia = B.rail_index[b][a], ic = B.rail_index[b][c];
  if (ia < 0 || ic < 0) return 0;
  return (B.straight[b][ia] >> ic) & 1;
}

JQ_API void* jq_game_new(const RuleConfig* cfg) { return new Game(cfg ? *cfg : RuleConfig()); }
JQ_API void jq_game_free(void* h) { delete G(h); }
JQ_API void jq_game_reset(void* h, uint64_t seed, const int8_t* setups, const int32_t* kinds) {
  int k[kSeats];
  if (kinds)
    for (int s = 0; s < kSeats; ++s) k[s] = kinds[s];
  G(h)->Reset(seed, setups, kinds ? k : nullptr);
}
JQ_API int jq_game_load_position(void* h, const int8_t* types, const int8_t* owners, int current_seat,
                                 uint64_t seed) {
  return G(h)->LoadPosition(types, owners, current_seat, seed);
}
JQ_API int jq_game_current_seat(void* h) { return G(h)->current_seat(); }
JQ_API int jq_game_over(void* h) { return G(h)->over(); }
JQ_API int jq_game_winner(void* h) { return G(h)->winner(); }
JQ_API int jq_game_end_reason(void* h) { return G(h)->end_reason(); }
JQ_API int jq_game_ply(void* h) { return G(h)->ply(); }
JQ_API int jq_game_team_of(void* h, int seat) { return G(h)->team_of(seat); }
JQ_API int jq_game_seat_alive(void* h, int seat) { return G(h)->active(seat) && G(h)->seat_alive(seat); }
JQ_API int jq_game_flag_revealed(void* h, int seat) { return G(h)->flag_revealed(seat); }

JQ_API int jq_game_legal_moves(void* h, int16_t* buf, int cap) {
  std::vector<Move> mv(size_t(cap > 0 ? cap : 1));
  const int n = G(h)->LegalMoves(mv.data(), cap);
  for (int i = 0; i < n && i < cap; ++i) {
    buf[2 * i] = mv[i].from;
    buf[2 * i + 1] = mv[i].to;
  }
  return n;
}
JQ_API int jq_game_step(void* h, int from, int to) { return G(h)->Step(from, to); }

JQ_API void jq_game_encode(void* h, int observer, float* node_feats, float* global, uint8_t* legal) {
  EncodeObservation(*G(h), observer, node_feats, global, legal);
}
JQ_API void jq_game_true_types(void* h, int observer, int8_t* out) { EncodeTrueTypes(*G(h), observer, out); }

// 调试/可视化：绝对编号下每个节点的 (兵种, 归属)，空为 -1
JQ_API void jq_game_board(void* h, int8_t* types, int8_t* owners) {
  for (int n = 0; n < kNodes; ++n) {
    const int pid = G(h)->piece_at(n);
    types[n] = pid < 0 ? -1 : G(h)->piece(pid).type;
    owners[n] = pid < 0 ? -1 : G(h)->piece(pid).owner;
  }
}
JQ_API void jq_game_belief(void* h, int observer, uint16_t* out) {
  for (int n = 0; n < kNodes; ++n) {
    const int pid = G(h)->piece_at(n);
    out[n] = pid < 0 ? 0 : G(h)->belief(observer, pid);
  }
}
JQ_API void jq_game_setups(void* h, int8_t* out) { std::memcpy(out, G(h)->setups().data(), kSeats * kCampNodes); }
JQ_API int jq_game_history(void* h, int16_t* moves, int8_t* outcomes, int cap) {
  const auto& hist = G(h)->history();
  const auto& oc = G(h)->outcomes();
  const int n = int(hist.size());
  for (int i = 0; i < n && i < cap; ++i) {
    if (moves) {
      moves[2 * i] = hist[i].from;
      moves[2 * i + 1] = hist[i].to;
    }
    if (outcomes) outcomes[i] = oc[i];
  }
  return n;
}

JQ_API void jq_rng_seed(uint64_t* state, uint64_t seed) {
  Rng r(seed);
  std::memcpy(state, r.s, sizeof(r.s));
}
JQ_API int jq_agent_act(void* h, int kind, float param, uint64_t* rng_state, int16_t* out) {
  Rng r;
  std::memcpy(r.s, rng_state, sizeof(r.s));
  const Move m = AgentAct(kind, param, *G(h), r);
  std::memcpy(rng_state, r.s, sizeof(r.s));
  out[0] = m.from;
  out[1] = m.to;
  return m.from >= 0;
}

JQ_API int jq_valid_setup(const int8_t* camp30) { return ValidSetup(camp30); }
JQ_API void jq_generate_setup(int kind, uint64_t* rng_state, int8_t* camp30) {
  Rng r;
  std::memcpy(r.s, rng_state, sizeof(r.s));
  GenerateSetup(kind, r, camp30);
  std::memcpy(rng_state, r.s, sizeof(r.s));
}
JQ_API void jq_set_setup_params(int slot, const float* params) { SetSetupParams(slot, params); }
JQ_API int jq_resolve(int attacker, int defender) { return Resolve(attacker, defender); }

// 多线程自对弈，棋谱写入 path（append=1 追加）。返回写入字节数，失败 -1。
JQ_API int64_t jq_selfplay(const SelfPlaySpec* spec, int n_games, uint64_t seed, int threads, const char* path,
                           int append, int64_t* stats) {
  std::vector<GameRecord> recs;
  RunSelfPlay(*spec, n_games, seed, threads, path ? &recs : nullptr, stats);
  if (!path) return 0;
  std::vector<uint8_t> bytes;
  bytes.reserve(size_t(n_games) * 1024);
  for (const auto& r : recs) AppendRecordBytes(r, bytes);
  FILE* f = std::fopen(path, append ? "ab" : "wb");
  if (!f) return -1;
  const size_t w = std::fwrite(bytes.data(), 1, bytes.size(), f);
  std::fclose(f);
  return w == bytes.size() ? int64_t(w) : -1;
}
