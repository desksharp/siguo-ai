// 棋谱记录（.jqr 二进制）与多线程自对弈
#include <atomic>
#include <cstring>
#include <thread>

#include "junqi/junqi.h"

namespace junqi {

GameRecord MakeRecord(const Game& g) {
  GameRecord r;
  r.cfg = g.cfg();
  r.seed = g.seed();
  r.setups = g.setups();
  r.moves = g.history();
  r.winner = int8_t(g.over() ? g.winner() : -1);
  r.end_reason = int8_t(g.end_reason());
  return r;
}

// 布局：magic"JQR1" | u8 ver,mode,vis,arc | u8 cmr,remove,reveal,corner | u16 max_plies,max_no_combat |
//       u64 seed | i8 winner | u8 end_reason | u16 nplies | i8 setups[120] | nplies×(u8 from,u8 to)
void AppendRecordBytes(const GameRecord& r, std::vector<uint8_t>& out) {
  auto put8 = [&](int v) { out.push_back(uint8_t(v)); };
  auto put16 = [&](int v) {
    put8(v & 0xFF);
    put8((v >> 8) & 0xFF);
  };
  out.insert(out.end(), {'J', 'Q', 'R', '1'});
  put8(1);
  put8(r.cfg.mode);
  put8(r.cfg.visibility);
  put8(r.cfg.arc_mode);
  put8(r.cfg.center_middle_rail);
  put8(r.cfg.remove_eliminated);
  put8(r.cfg.reveal_flag_on_marshal_death);
  put8(r.cfg.camp_corner_straight);
  put16(r.cfg.max_plies);
  put16(r.cfg.max_plies_without_combat);
  for (int k = 0; k < 8; ++k) put8(int((r.seed >> (8 * k)) & 0xFF));
  put8(r.winner);
  put8(r.end_reason);
  put16(int(r.moves.size()));
  for (int8_t t : r.setups) put8(t);
  for (const Move& m : r.moves) {
    put8(m.from);
    put8(m.to);
  }
}

void RunSelfPlay(const SelfPlaySpec& spec, int n_games, uint64_t seed, int threads, std::vector<GameRecord>* out,
                 int64_t* stats) {
  if (threads < 1) threads = 1;
  if (out) out->assign(size_t(n_games), GameRecord{});
  std::atomic<int> next{0};
  std::atomic<int64_t> wins0{0}, wins1{0}, draws{0}, plies{0};
  auto worker = [&] {
    Game g(spec.cfg);
    for (;;) {
      const int i = next.fetch_add(1);
      if (i >= n_games) break;
      uint64_t x = seed + uint64_t(i) * 0x9E3779B97F4A7C15ull;
      const uint64_t game_seed = SplitMix64(x);
      g.Reset(game_seed, nullptr, spec.setup_kind);
      Rng rng(game_seed ^ 0xA5A5A5A5DEADBEEFull);
      while (!g.over()) {
        const int s = g.current_seat();
        const Move m = AgentAct(spec.agent_kind[s], spec.agent_param[s], g, rng);
        if (g.Step(m.from, m.to) == kOutIllegal) break;  // 不应发生，测试会兜住
      }
      plies += g.ply();
      if (g.winner() == 0)
        ++wins0;
      else if (g.winner() == 1)
        ++wins1;
      else
        ++draws;
      if (out) (*out)[size_t(i)] = MakeRecord(g);
    }
  };
  std::vector<std::thread> pool;
  for (int t = 1; t < threads; ++t) pool.emplace_back(worker);
  worker();
  for (auto& th : pool) th.join();
  if (stats) {
    stats[0] = wins0;
    stats[1] = wins1;
    stats[2] = draws;
    stats[3] = plies;
  }
}

}  // namespace junqi
