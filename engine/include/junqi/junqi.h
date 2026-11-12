// 四国军旗模拟引擎 · 核心头文件
//
// 节点编号（固定，全项目通用）：
//   营区节点 id = seat*30 + row*5 + col   (seat 0..3, row 0=前排..5=底排, col 0..4 从该方视角左→右)
//   九宫节点 id = 120 + j*3 + i           (17×17 网格坐标 x=6+2i, y=6+2j)
// 座位：0=南 1=东 2=北 3=西，行棋顺序 0→1→2→3；四国模式 {0,2} vs {1,3}，二人模式只有 0 与 2。
// 军衔编码 0..11：军旗 地雷 炸弹 工兵 排长 连长 营长 团长 旅长 师长 军长 司令。
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace junqi {

constexpr int kNodes = 129;
constexpr int kSeats = 4;
constexpr int kCampNodes = 30;
constexpr int kPiecesPerSeat = 25;
constexpr int kPieces = kSeats * kPiecesPerSeat;
constexpr int kTypes = 12;
constexpr int kCenterBase = 120;
constexpr int kGrid = 17;

enum PieceType : int {
  kFlag = 0,
  kMine,
  kBomb,
  kEngineer,
  kPlatoon,
  kCompany,
  kBattalion,
  kRegiment,
  kBrigade,
  kDivision,
  kCorps,
  kMarshal,
};
inline constexpr std::array<int, kTypes> kQuota = {1, 3, 2, 3, 3, 3, 2, 2, 2, 2, 1, 1};

// 0=移动 1=吃掉 2=被吃 3=同归于尽 4=夺旗
enum Outcome : int { kOutMove = 0, kOutKill = 1, kOutKilled = 2, kOutBoth = 3, kOutCaptureFlag = 4, kOutIllegal = -1 };
enum Mode : int { kModeFour = 0, kModeDuel = 1 };
enum Visibility : int { kVisDark = 0, kVisTeam = 1, kVisAll = 2 };
// 弧形铁路（相邻两家前排角点之间）对非工兵的直行规则
enum ArcMode : int { kArcNone = 0, kArcFrontRows = 1, kArcSideRails = 2, kArcBoth = 3 };
enum EndReason : int { kEndNone = 0, kEndTeamEliminated = 1, kEndMaxPlies = 2, kEndNoCombat = 3 };
// kSetupHeuristic：现行布阵，参数来自布阵搜索（rules.cpp kSearchedSetup）
// kSetupLegacy：2026-10-31 之前的启发式（另一个大本营可能放大子、工兵、炸弹），留作对照
// kSetupCustom + k：第 k 组参数化布阵（SetSetupParams 设置，布阵搜索用）
enum SetupKind : int { kSetupRandom = 0, kSetupHeuristic = 1, kSetupLegacy = 2, kSetupCustom = 3 };
inline constexpr int kSetupSlots = 8;
// 参数（都是相对权重，列按“军旗在第 1 列”镜像）：另一个大本营放 雷/排长/连长 [3]、
// 地雷落点（第 4 行 5 列 + 第 5 行非大本营 3 列）[8]、炸弹…司令 10 种子的行权重 [10×6] 和列权重 [10×5]
inline constexpr int kSetupParams = 3 + 8 + 60 + 50;

using Mask = uint16_t;
constexpr Mask kAllTypes = 0x0FFF;
constexpr Mask Bit(int t) { return Mask(1u << t); }
int Popcount(Mask m);

// 与 C API 共用的 POD 配置（全部 int32，便于 ctypes 映射）
struct RuleConfig {
  int32_t mode = kModeFour;
  int32_t visibility = kVisDark;
  int32_t arc_mode = kArcSideRails;  // 弧线只连通两家的边线铁路（主流平台规则）
  int32_t center_middle_rail = 1;  // 营区前排中点 ↔ 九宫 是铁路
  int32_t remove_eliminated = 1;   // 出局方棋子移出棋盘
  int32_t reveal_flag_on_marshal_death = 1;
  int32_t max_plies = 2000;
  int32_t max_plies_without_combat = 400;
  int32_t camp_corner_straight = 0;  // 1 = 营区边线 ↔ 第5排的直角拐角视为直行；默认 0：直角都算拐弯，只有弧线（圆角）能直行
};

inline int CampNode(int seat, int row, int col) { return seat * kCampNodes + row * 5 + col; }
inline bool IsCampCell(int row, int col) {  // 行营
  return (row == 1 && (col == 1 || col == 3)) || (row == 2 && col == 2) || (row == 3 && (col == 1 || col == 3));
}
inline bool IsBaseCell(int row, int col) { return row == 5 && (col == 1 || col == 3); }  // 大本营

// ---------------------------------------------------------------- Board
struct Board {
  int arc_mode = kArcSideRails;
  bool center_middle_rail = true;
  bool camp_corner_straight = false;

  std::array<int8_t, kNodes> seat_of{};  // 九宫为 -1
  std::array<int8_t, kNodes> row{}, col{};
  std::array<int8_t, kNodes> x{}, y{};
  std::array<uint8_t, kNodes> is_rail{}, is_camp{}, is_base{};
  std::array<int16_t, kGrid * kGrid> node_at{};

  std::array<std::vector<int16_t>, kNodes> adj;       // 一步可达（公路+铁路+行营斜线）
  std::array<std::vector<int16_t>, kNodes> rail_adj;  // 铁路边
  // straight[b][i]：从 rail_adj[b][i] 走到 b 后，可以“直行”继续到哪些 rail_adj[b][j]（位掩码）
  std::array<std::vector<uint8_t>, kNodes> straight;
  std::array<std::array<int8_t, kNodes>, kNodes> rail_index{};  // rail_index[b][a] = a 在 rail_adj[b] 中的下标
  std::array<std::array<uint8_t, kNodes>, kNodes> edge_kind{};  // 0 无 1 公路 2 铁路
  std::array<std::array<uint8_t, kNodes>, kNodes> dist{};       // 无阻挡最短步数
  std::array<std::array<int16_t, kNodes>, kSeats> rel{};        // rel[o][abs]：以 o 为 0 号座位的相对编号
  std::array<std::array<int16_t, kNodes>, kSeats> abs_of{};

  Board(int arc_mode, bool center_middle_rail, bool camp_corner_straight);
  static const Board& Get(int arc_mode, bool center_middle_rail, bool camp_corner_straight);
  static const Board& Get(const RuleConfig& cfg) {
    return Get(cfg.arc_mode, cfg.center_middle_rail != 0, cfg.camp_corner_straight != 0);
  }
};

// ---------------------------------------------------------------- RNG
struct Rng {
  uint64_t s[4];
  explicit Rng(uint64_t seed = 0) { Seed(seed); }
  void Seed(uint64_t seed);
  uint64_t Next();
  double Uniform() { return (Next() >> 11) * (1.0 / 9007199254740992.0); }
  int Below(int n) { return int(Next() % uint64_t(n)); }
};
uint64_t SplitMix64(uint64_t& x);

// ---------------------------------------------------------------- Rules
int Resolve(int attacker, int defender);  // 返回 Outcome
// 与某结局相容的对手兵种集合
Mask CompatDefender(int outcome, int attacker_type);
Mask CompatAttacker(int outcome, int defender_type);
bool ValidSetup(const int8_t* camp30);
void GenerateSetup(int kind, Rng& rng, int8_t* camp30);
void SetSetupParams(int slot, const float* params);

struct Move {
  int16_t from;
  int16_t to;
};

struct Piece {
  int8_t owner = -1;
  int8_t type = -1;
  int16_t node = -1;
  uint8_t alive = 0;
  uint8_t moved = 0;
  int16_t combats = 0;
  int32_t last_moved_ply = -1;
  int16_t start_node = -1;
};

// ---------------------------------------------------------------- Game
class Game {
 public:
  explicit Game(const RuleConfig& cfg = RuleConfig());

  // setups: kSeats*30 个兵种（行营与未参战座位为 -1），nullptr 则按 setup_kinds 生成
  void Reset(uint64_t seed, const int8_t* setups = nullptr, const int* setup_kinds = nullptr);
  // 载入任意局面（残局课程、对拍）。types/owners 为绝对节点上的兵种与归属，-1 为空。
  // 载入局面没有布阵记录、各方子数可以不足 25，信念只做在“子数不全”下仍成立的推断。
  bool LoadPosition(const int8_t* types, const int8_t* owners, int current_seat, uint64_t seed);

  const RuleConfig& cfg() const { return cfg_; }
  const Board& board() const { return *B_; }
  int current_seat() const { return cur_; }
  bool over() const { return over_; }
  int winner() const { return winner_; }  // 胜方队伍 0/1，和棋 -1
  int end_reason() const { return end_reason_; }
  int ply() const { return ply_; }
  int plies_since_combat() const { return plies_since_combat_; }
  uint64_t seed() const { return seed_; }
  int team_of(int seat) const { return team_[seat]; }
  bool active(int seat) const { return active_[seat]; }
  bool seat_alive(int seat) const { return alive_[seat]; }
  bool flag_revealed(int seat) const { return flag_revealed_[seat]; }
  int pieces_alive(int seat) const;
  int piece_at(int node) const { return board_[node]; }
  const Piece& piece(int pid) const { return pieces_[pid]; }
  Mask belief(int observer, int pid) const { return belief_[observer][pid]; }
  bool visible(int observer, int owner) const;
  const std::vector<Move>& history() const { return history_; }
  const std::vector<int8_t>& outcomes() const { return outcomes_; }
  const std::array<int8_t, kSeats * kCampNodes>& setups() const { return setups_; }

  int GenMoves(int seat, Move* out, int cap) const;  // 返回总数（可能 > cap）
  // 只用 observer 掌握的信息推算 seat 的可达着（不读对 observer 隐藏的真实兵种）：
  // 信念里还可能是可动兵种 → 按可动算；还可能是工兵 → 按工兵算（工兵可达是直行可达的超集）
  int GenMovesBelief(int seat, int observer, Move* out, int cap) const;
  int LegalMoves(Move* out, int cap) const { return GenMoves(cur_, out, cap); }
  bool IsLegal(int from, int to) const;
  // 该子从 from 出发、按非工兵规则（一步或直行）能否到达 to
  bool StraightReach(int pid, int to) const;
  int Step(int from, int to);  // 返回 Outcome；非法着返回 kOutIllegal 且不改变局面

 private:
  template <class Probe>
  void Slide(int from, bool engineer, Probe&& probe) const;
  void AddPieceMoves(int pid, bool mobile, bool engineer, Move* out, int& n, int cap) const;
  void Eliminate(int seat);
  void Advance();
  void InitBelief(bool from_setup);
  void PublicMoved(int pid);
  void PublicSetMask(int pid, Mask m);
  void PublicCombat(int a, int d, int outcome);
  void Propagate(int observer, int owner);

  RuleConfig cfg_;
  const Board* B_;
  uint64_t seed_ = 0;
  std::array<int16_t, kNodes> board_{};
  std::array<Piece, kPieces> pieces_{};
  std::array<std::array<Mask, kPieces>, kSeats> belief_{};
  std::array<int8_t, kSeats * kCampNodes> setups_{};
  std::array<int8_t, kSeats> team_{}, active_{}, alive_{}, flag_revealed_{};
  int cur_ = 0, ply_ = 0, plies_since_combat_ = 0, winner_ = -1, end_reason_ = kEndNone;
  bool over_ = false;
  bool roster_complete_ = true;
  std::vector<Move> history_;
  std::vector<int8_t> outcomes_;

  mutable std::array<uint32_t, kNodes> dest_stamp_{}, visit_stamp_{};
  mutable std::array<std::array<uint32_t, 6>, kNodes> edge_stamp_{};
  mutable uint32_t dest_gen_ = 0, visit_gen_ = 0, edge_gen_ = 0;
};

// ---------------------------------------------------------------- Observation encoding
constexpr int kNodeFeatures = 41;
constexpr int kGlobalFeatures = 17;
// node_feats [kNodes*kNodeFeatures]、global [kGlobalFeatures]、legal_mask [kNodes*kNodes]（可为 nullptr）
// 全部以 observer 为 0 号座位旋转后的相对编号。
void EncodeObservation(const Game& g, int observer, float* node_feats, float* global, uint8_t* legal_mask);
// 特权信息（仅训练）：相对编号下每个节点上棋子的真实兵种，空为 -1
void EncodeTrueTypes(const Game& g, int observer, int8_t* out);

// ---------------------------------------------------------------- Agents
// kAgentExploit：规则 AI 的变体，认为对方后两排没动过的子多半是地雷，专打动过（已排除地雷）的后排子
enum AgentKind : int { kAgentRandom = 0, kAgentRule = 1, kAgentExploit = 2 };
Move RandomAct(const Game& g, Rng& rng);
Move RuleAct(const Game& g, Rng& rng, float temperature, float mine_boost = 1.0f);
Move AgentAct(int kind, float param, const Game& g, Rng& rng);

// ---------------------------------------------------------------- Records & self-play
struct GameRecord {
  RuleConfig cfg;
  uint64_t seed = 0;
  std::array<int8_t, kSeats * kCampNodes> setups{};
  std::vector<Move> moves;
  int8_t winner = -1;
  int8_t end_reason = kEndNone;
};
GameRecord MakeRecord(const Game& g);
void AppendRecordBytes(const GameRecord& r, std::vector<uint8_t>& out);

struct SelfPlaySpec {
  RuleConfig cfg;
  int32_t agent_kind[kSeats] = {kAgentRule, kAgentRule, kAgentRule, kAgentRule};
  float agent_param[kSeats] = {0.1f, 0.1f, 0.1f, 0.1f};
  int32_t setup_kind[kSeats] = {kSetupHeuristic, kSetupHeuristic, kSetupHeuristic, kSetupHeuristic};
};
// stats: [队伍0胜, 队伍1胜, 和棋, 总步数]
void RunSelfPlay(const SelfPlaySpec& spec, int n_games, uint64_t seed, int threads, std::vector<GameRecord>* out,
                 int64_t* stats);

}  // namespace junqi
