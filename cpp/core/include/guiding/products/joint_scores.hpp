#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "guiding/table/frame.hpp"

// Port of cap_guiding/joint_scores.py: guiding triplet scores joined with
// channel-vs-uniform beamlike scores.
namespace guiding::products {

enum class JoinHow { Inner, Left, Right, Outer };
[[nodiscard]] JoinHow parse_join_how(std::string_view name);

// pandas.merge(left, right, on=keys, how=how) for str key columns: left
// columns, then the right columns other than the keys. inner/left keep the
// left order, right the right order, outer sorts the keys.
[[nodiscard]] table::Frame merge_frames(const table::Frame& left, const table::Frame& right,
                                        const std::vector<std::string>& keys, JoinHow how);

// load_and_join_guiding_beamlike (throws for missing join keys).
[[nodiscard]] table::Frame load_and_join_guiding_beamlike(const std::filesystem::path& triplet_scores_csv,
                                                          const std::filesystem::path& beamlike_pair_scores_csv,
                                                          JoinHow how);

[[nodiscard]] table::Frame compute_joint_correlations(const table::Frame& joined);
// value_counts of joint_bucket / triple_bucket: descending counts, ties by first appearance.
[[nodiscard]] table::Frame bucket_counts(const table::Frame& joined, const std::string& column);

// numpy.corrcoef(x, y)[0, 1] and scipy.stats.spearmanr(x, y)[0] (average ranks).
[[nodiscard]] double pearson(std::span<const double> x, std::span<const double> y);
[[nodiscard]] double spearman(std::span<const double> x, std::span<const double> y);

struct JointOutput {
  std::string key;
  std::filesystem::path path;
};

// write_joint_outputs: every CSV of the reference, in its order.
std::vector<JointOutput> write_joint_outputs(const std::filesystem::path& outdir, const table::Frame& joined,
                                             std::size_t top);

// Rows of `joined` selected by `keep`, sorted by `column` descending, first `top`.
[[nodiscard]] table::Frame sorted_subset(const table::Frame& joined, const std::vector<std::size_t>& rows,
                                         const std::string& column, bool ascending, std::size_t top);

}  // namespace guiding::products
