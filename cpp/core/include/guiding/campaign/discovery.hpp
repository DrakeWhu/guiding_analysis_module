#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

// Port of cap_guiding/campaign.py: case discovery, readiness, triplets and the
// campaign report CSVs.
namespace guiding::campaign {

enum class CaseType { Channel, Uniform, Vacuum };

[[nodiscard]] const char* case_type_name(CaseType type) noexcept;

using Token = std::optional<std::string>;

// Normalised name tokens ("." -> "p", lower case), as campaign.py extracts them.
struct CaseTokens {
  Token laser_case;
  Token density;
  Token ref_density;
  Token plateau;
  Token focus;
  Token diameter;
};

struct CaseInfo {
  std::string case_id;
  CaseType case_type = CaseType::Channel;
  std::filesystem::path case_dir;
  std::filesystem::path diag_dir;
  std::int64_t h5_count = 0;
  // Newest *.h5 mtime, scanned once at discovery (Python rescans on every
  // age query; caching avoids repeated recursive listings on GPFS/Lustre).
  std::optional<double> newest_h5_mtime_s;
  CaseTokens tokens;
  std::string source;

  using BaseKey = std::tuple<Token, Token, Token>;
  using FullKey = std::tuple<Token, Token, Token, Token>;

  [[nodiscard]] Token density_key() const {
    return case_type == CaseType::Vacuum ? tokens.ref_density : tokens.density;
  }
  [[nodiscard]] BaseKey base_key() const { return {tokens.laser_case, tokens.plateau, tokens.focus}; }
  [[nodiscard]] FullKey full_key() const {
    return {tokens.laser_case, tokens.plateau, tokens.focus, density_key()};
  }
};

struct TripletInfo {
  CaseInfo::FullKey key;
  std::optional<CaseInfo> channel;
  std::optional<CaseInfo> uniform;
  std::optional<CaseInfo> vacuum;

  [[nodiscard]] bool complete() const { return channel && uniform && vacuum; }
  // f<fnum>_n<density>_L<plateau>mm_d<diameter>um_foc<focus>, missing parts skipped.
  [[nodiscard]] std::string label() const;
};

[[nodiscard]] std::optional<CaseType> infer_case_type(std::string_view case_id);
[[nodiscard]] CaseTokens parse_case_tokens(std::string_view case_id);

// CASE/diags/fields, else CASE/diags/diag1; with require_exists=false the
// fields path is returned when neither exists (diagnostics.py).
[[nodiscard]] std::filesystem::path resolve_field_diag_dir(const std::filesystem::path& case_dir,
                                                           bool require_exists = true);

struct H5Scan {
  std::int64_t count = 0;
  std::optional<double> newest_mtime_s;
};

// Regular *.h5 files anywhere below diag_dir (campaign.py:iter_h5_files).
[[nodiscard]] H5Scan scan_h5_files(const std::filesystem::path& diag_dir);

// time.time()
[[nodiscard]] double unix_time_now();
// max(0, (now - newest mtime) / 60); nullopt without HDF5 files.
[[nodiscard]] std::optional<double> newest_h5_age_min(const CaseInfo& info, double now_s);

[[nodiscard]] std::optional<CaseInfo> infer_case_info_from_dir(const std::filesystem::path& case_dir,
                                                               const std::string& source = "name");
[[nodiscard]] std::vector<CaseInfo> load_cases_from_cases_full(const std::filesystem::path& campaign_root);
[[nodiscard]] std::vector<CaseInfo> discover_cases_by_name(const std::filesystem::path& campaign_root);
// cases_full.tsv / cases.tsv when present and non-empty, else directory names.
[[nodiscard]] std::vector<CaseInfo> discover_cases(const std::filesystem::path& campaign_root);
[[nodiscard]] std::vector<TripletInfo> build_triplets(const std::vector<CaseInfo>& cases);

[[nodiscard]] bool case_has_min_h5(const CaseInfo& info, std::int64_t min_h5);
[[nodiscard]] bool case_has_stable_h5(const CaseInfo& info, double min_last_h5_age_min, double now_s);
[[nodiscard]] bool case_is_ready(const CaseInfo& info, std::int64_t min_h5, double min_last_h5_age_min, double now_s);
[[nodiscard]] bool triplet_ready_min_h5(const TripletInfo& triplet, std::int64_t min_h5);
[[nodiscard]] bool triplet_is_ready(const TripletInfo& triplet, std::int64_t min_h5, double min_last_h5_age_min,
                                    double now_s);

struct CampaignReportPaths {
  std::filesystem::path cases;
  std::filesystem::path triplets;
  std::filesystem::path insufficient_h5;
  std::filesystem::path unstable_h5;
};

// campaign_cases.csv, campaign_triplets.csv, campaign_insufficient_h5.csv and
// campaign_unstable_h5.csv, as csv.DictWriter writes them.
CampaignReportPaths write_campaign_report(const std::vector<CaseInfo>& cases, const std::vector<TripletInfo>& triplets,
                                          const std::filesystem::path& outdir, std::int64_t min_h5,
                                          double min_last_h5_age_min, double now_s);

}  // namespace guiding::campaign
