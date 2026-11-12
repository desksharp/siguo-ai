// 引擎自测：棋盘不变量、碰子表、布阵约束、随机对局模糊测试（含信念可靠性）、确定性
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>

#include "junqi/junqi.h"

using namespace junqi;

static int g_fail = 0;
#define CHECK(cond)                                                     \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      ++g_fail;                                                         \
    }                                                                   \
  } while (0)

static void TestBoard() {
  for (int variant = 0; variant < 16; ++variant) {
    {
      const int arc = variant & 3, cmr = (variant >> 2) & 1, corner = (variant >> 3) & 1;
      const Board& B = Board::Get(arc, cmr != 0, corner != 0);
      std::set<std::pair<int, int>> xy;
      int camps = 0, bases = 0;
      for (int n = 0; n < kNodes; ++n) {
        xy.insert({B.x[n], B.y[n]});
        camps += B.is_camp[n];
        bases += B.is_base[n];
        CHECK(B.node_at[B.y[n] * kGrid + B.x[n]] == n);
        for (int m = 0; m < kNodes; ++m) {
          CHECK(B.edge_kind[n][m] == B.edge_kind[m][n]);
          // 旋转对称
          CHECK(B.edge_kind[n][m] == B.edge_kind[B.rel[1][n]][B.rel[1][m]]);
          CHECK(B.dist[n][m] < 255);
        }
        CHECK(B.abs_of[0][n] == n);
        for (int o = 0; o < kSeats; ++o) CHECK(B.abs_of[o][B.rel[o][n]] == n);
      }
      CHECK(int(xy.size()) == kNodes);
      CHECK(camps == 20);
      CHECK(bases == 8);
    }
  }
  const Board& B = Board::Get(kArcBoth, true, true);
  auto at = [&](int x, int y) { return int(B.node_at[y * kGrid + x]); };
  // 规则要求的连接：弧线、贯通线、中路铁路
  CHECK(B.edge_kind[at(10, 11)][at(11, 10)] == 2);
  CHECK(B.edge_kind[at(8, 11)][at(8, 10)] == 2);
  CHECK(B.edge_kind[at(6, 11)][at(6, 10)] == 2);
  CHECK(B.edge_kind[at(7, 11)][at(7, 12)] == 1);   // 营区内公路
  CHECK(B.edge_kind[at(7, 12)][at(8, 13)] == 1);   // 行营斜线
  CHECK(B.edge_kind[at(7, 11)][at(7, 10)] == 0);   // 山界不通
  CHECK(B.edge_kind[at(6, 15)][at(6, 16)] == 1);   // 底排是公路
  CHECK(B.is_camp[at(8, 13)] && B.is_base[at(7, 16)] && B.is_base[at(9, 16)]);
  // 九宫中心 4 条铁路
  CHECK(B.rail_adj[at(8, 8)].size() == 4);
  // 弧线（默认 side_rails）：边线 → 弧线 → 相邻家边线算直行；前排 → 弧线 → 相邻家前排不算
  const RuleConfig def_cfg{};
  const Board& D = Board::Get(def_cfg);
  auto straight = [&](int a, int m, int c) {
    const int ia = D.rail_index[m][a], ic = D.rail_index[m][c];
    return ia >= 0 && ic >= 0 && ((D.straight[m][ia] >> ic) & 1) != 0;
  };
  CHECK(D.arc_mode == kArcSideRails);
  CHECK(straight(at(6, 12), at(6, 11), at(5, 10)));   // 南左边线 → 弧线
  CHECK(straight(at(6, 11), at(5, 10), at(4, 10)));   // 弧线 → 西边线
  CHECK(!straight(at(5, 9), at(5, 10), at(6, 11)));   // 西前排 → 弧线
  CHECK(!straight(at(5, 10), at(6, 11), at(7, 11)));  // 弧线 → 南前排
  CHECK(!D.camp_corner_straight);
  CHECK(!straight(at(6, 14), at(6, 15), at(7, 15)));   // 南左边线 → 第5排：直角，算拐弯
  CHECK(!straight(at(10, 14), at(10, 15), at(9, 15))); // 南右边线 → 第5排
  CHECK(!straight(at(2, 10), at(1, 10), at(1, 9)));    // 西边线 → 西第5排
}

static void TestRules() {
  CHECK(Resolve(kMarshal, kCorps) == kOutKill);
  CHECK(Resolve(kCorps, kMarshal) == kOutKilled);
  CHECK(Resolve(kDivision, kDivision) == kOutBoth);
  CHECK(Resolve(kBomb, kMarshal) == kOutBoth);
  CHECK(Resolve(kEngineer, kBomb) == kOutBoth);
  CHECK(Resolve(kEngineer, kMine) == kOutKill);
  CHECK(Resolve(kMarshal, kMine) == kOutKilled);
  CHECK(Resolve(kBomb, kMine) == kOutBoth);
  CHECK(Resolve(kPlatoon, kFlag) == kOutCaptureFlag);
  CHECK(Resolve(kMine, kPlatoon) == kOutIllegal);
  CHECK(Resolve(kPlatoon, kEngineer) == kOutKill);
  CHECK((CompatDefender(kOutKilled, kDivision) & Bit(kMine)) != 0);
  CHECK((CompatDefender(kOutKilled, kDivision) & Bit(kBrigade)) == 0);
}

static void TestSetups() {
  Rng rng(7);
  for (int i = 0; i < 20000; ++i) {
    int8_t camp[kCampNodes];
    GenerateSetup(i % 3, rng, camp);
    CHECK(ValidSetup(camp));
    if (i % 3 == kSetupHeuristic) {  // 不放军旗的那个大本营只放地雷、排长、连长；最后一排没有工兵
      const int other = camp[5 * 5 + 1] == kFlag ? camp[5 * 5 + 3] : camp[5 * 5 + 1];
      CHECK(other == kMine || other == kPlatoon || other == kCompany);
      for (int c = 0; c < 5; ++c) CHECK(camp[5 * 5 + c] != kEngineer);
    }
  }
  {  // 参数化布阵：极端权重也要合法
    float params[kSetupParams];
    for (float& v : params) v = 1.0f;
    params[2] = 0.0f;  // 大本营不放连长
    SetSetupParams(0, params);
    for (int i = 0; i < 5000; ++i) {
      int8_t camp[kCampNodes];
      GenerateSetup(kSetupCustom, rng, camp);
      CHECK(ValidSetup(camp));
      const int other = camp[5 * 5 + 1] == kFlag ? camp[5 * 5 + 3] : camp[5 * 5 + 1];
      CHECK(other == kMine || other == kPlatoon);
    }
  }
  int8_t bad[kCampNodes];
  GenerateSetup(kSetupRandom, rng, bad);
  for (int pos = 0; pos < 5; ++pos)
    if (bad[pos] != kBomb) {
      for (int p2 = 5; p2 < kCampNodes; ++p2)
        if (bad[p2] == kBomb) {
          std::swap(bad[pos], bad[p2]);
          break;
        }
      break;
    }
  CHECK(!ValidSetup(bad));
}

// 随机/规则对局：每一步检查局面一致性与“信念掩码必含真值”
static void FuzzGames(int mode, int vis, int games, int agent) {
  RuleConfig cfg;
  cfg.mode = mode;
  cfg.visibility = vis;
  Game g(cfg);
  Rng rng(12345 + mode * 10 + vis);
  int illegal = 0;
  for (int gi = 0; gi < games; ++gi) {
    g.Reset(1000 + gi);
    while (!g.over()) {
      Move buf[1024];
      const int n = g.LegalMoves(buf, 1024);
      CHECK(n > 0 && n <= 1024);
      for (int i = 0; i < n; ++i) {
        const int pid = g.piece_at(buf[i].from);
        CHECK(pid >= 0 && g.piece(pid).owner == g.current_seat());
        const int d = g.piece_at(buf[i].to);
        if (d >= 0) CHECK(g.team_of(g.piece(d).owner) != g.team_of(g.current_seat()));
      }
      const Move m = AgentAct(agent, 0.2f, g, rng);
      if (g.Step(m.from, m.to) == kOutIllegal) ++illegal;
      int on_board = 0;
      for (int nd = 0; nd < kNodes; ++nd) {
        const int pid = g.piece_at(nd);
        if (pid < 0) continue;
        ++on_board;
        CHECK(g.piece(pid).alive && g.piece(pid).node == nd);
      }
      int alive = 0;
      for (int pid = 0; pid < kPieces; ++pid) {
        const Piece& p = g.piece(pid);
        alive += p.alive;
        if (p.type < 0) continue;
        for (int o = 0; o < kSeats; ++o) {
          if (!g.active(o)) continue;
          CHECK((g.belief(o, pid) & Bit(p.type)) != 0);
        }
      }
      CHECK(alive == on_board);
    }
    CHECK(g.end_reason() != kEndNone);
  }
  CHECK(illegal == 0);
}

static void TestDeterminism() {
  RuleConfig cfg;
  SelfPlaySpec spec;
  spec.cfg = cfg;
  std::vector<GameRecord> a, b;
  int64_t sa[4], sb[4];
  RunSelfPlay(spec, 16, 99, 4, &a, sa);
  RunSelfPlay(spec, 16, 99, 1, &b, sb);
  for (int i = 0; i < 16; ++i) {
    CHECK(a[i].moves.size() == b[i].moves.size());
    CHECK(std::memcmp(a[i].setups.data(), b[i].setups.data(), 120) == 0);
    for (size_t k = 0; k < a[i].moves.size() && k < b[i].moves.size(); ++k)
      CHECK(a[i].moves[k].from == b[i].moves[k].from && a[i].moves[k].to == b[i].moves[k].to);
  }
  // 复盘：用 setups + 着法序列重放得到同样的结局
  for (int i = 0; i < 16; ++i) {
    Game g(cfg);
    g.Reset(a[i].seed, a[i].setups.data());
    for (const Move& m : a[i].moves) CHECK(g.Step(m.from, m.to) != kOutIllegal);
    CHECK(g.over() && g.winner() == a[i].winner);
  }
}

int main() {
  TestBoard();
  TestRules();
  TestSetups();
  FuzzGames(kModeFour, kVisDark, 300, kAgentRandom);
  FuzzGames(kModeFour, kVisTeam, 100, kAgentRandom);
  FuzzGames(kModeDuel, kVisDark, 300, kAgentRandom);
  FuzzGames(kModeFour, kVisDark, 60, kAgentRule);
  FuzzGames(kModeDuel, kVisDark, 60, kAgentRule);
  TestDeterminism();
  if (g_fail) {
    std::printf("FAILED: %d checks\n", g_fail);
    return 1;
  }
  std::printf("all engine tests passed\n");
  return 0;
}
