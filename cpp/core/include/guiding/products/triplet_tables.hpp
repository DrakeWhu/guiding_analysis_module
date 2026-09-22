#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "guiding/table/frame.hpp"

// Port of cap_guiding/triplet.py (channel / uniform / vacuum comparison tables).
namespace guiding::products {

// triplet.REQUIRED_COLUMNS: what a valid guiding_metrics.csv must provide.
[[nodiscard]] const std::vector<std::string>& triplet_required_columns();

struct TripletTables {
  table::Frame long_table;
  table::Frame wide;
  table::Frame late_summary;
  table::Frame late_ratios;
};

// build_triplet_tables(channel_csv, uniform_csv, vacuum_csv, label, late_fraction)
[[nodiscard]] TripletTables build_triplet_tables(const std::filesystem::path& channel_csv,
                                                 const std::filesystem::path& uniform_csv,
                                                 const std::filesystem::path& vacuum_csv,
                                                 const std::string& label = "triplet",
                                                 double late_fraction = 1.0 / 3.0);

struct TripletTablePaths {
  std::filesystem::path long_table;
  std::filesystem::path wide;
  std::filesystem::path late_summary;
  std::filesystem::path late_ratios;
};

// guiding_triplet_{long,wide,late_summary,late_ratios}.csv in outdir.
TripletTablePaths write_triplet_tables(const TripletTables& tables, const std::filesystem::path& outdir);

}  // namespace guiding::products
