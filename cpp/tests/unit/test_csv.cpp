#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "guiding/products/singlecase_score.hpp"
#include "guiding/table/csv.hpp"
#include "guiding/table/py_format.hpp"
#include "guiding/table/record.hpp"
#include "support/table_compare.hpp"

using guiding::table::Cell;
using guiding::table::CsvDialect;

TEST_CASE("csv.DictWriter dialect: CRLF, repr floats, nan and minimal quoting", "[csv]") {
  std::string out;
  const std::vector<std::string> header{"a", "b", "c", "d", "e", "f"};
  guiding::table::append_csv_header(out, header, CsvDialect::python_csv());
  const std::vector<Cell> cells{std::int64_t{7},  2.0, std::numeric_limits<double>::quiet_NaN(), true,
                                std::string("x,y"), Cell{}};
  guiding::table::append_csv_record(out, cells, CsvDialect::python_csv());
  CHECK(out == "a,b,c,d,e,f\r\n7,2.0,nan,True,\"x,y\",\r\n");
}

TEST_CASE("pandas dialect writes NaN as an empty field and LF rows", "[csv]") {
  std::string out;
  const std::vector<Cell> cells{std::string("case"), std::numeric_limits<double>::quiet_NaN(), 1e-5};
  guiding::table::append_csv_record(out, cells, CsvDialect::pandas());
  CHECK(out == "case,,1e-05\n");
}

TEST_CASE("a lone empty field is quoted like csv.writer does", "[csv]") {
  std::string out;
  const std::vector<Cell> cells{std::string()};
  guiding::table::append_csv_record(out, cells, CsvDialect::pandas());
  CHECK(out == "\"\"\n");
}

TEST_CASE("CSV parsing handles quotes, CRLF and a BOM", "[csv]") {
  const auto table = guiding::table::parse_csv("\xEF\xBB\xBFid,name\r\n1,\"a \"\"b\"\", c\"\r\n2,plain\r\n");
  REQUIRE(table.columns == std::vector<std::string>{"id", "name"});
  REQUIRE(table.rows.size() == 2);
  CHECK(table.rows[0][1] == "a \"b\", c");
  CHECK(table.numeric_column("id") == std::vector<double>{1.0, 2.0});
  const auto names = table.numeric_column("name");
  CHECK(std::isnan(names[0]));
}

TEST_CASE("Record keeps dict insertion order when keys are reassigned", "[csv]") {
  guiding::table::Record record{{"status", std::string("failed")}, {"reason", std::string()}, {"score", 1.0}};
  record.set("extra", std::int64_t{3});
  record.set("reason", std::string("empty_csv"));
  const std::vector<guiding::table::Record> records{record};
  CHECK(guiding::table::format_records_pandas(records) == "status,reason,score,extra\nfailed,empty_csv,1.0,3\n");
}

TEST_CASE("python_path_string mirrors str(pathlib.Path)", "[csv]") {
  using guiding::table::python_path_string;
  CHECK(python_path_string("./out//case/") == "out/case");
  CHECK(python_path_string("/a/./b/") == "/a/b");
  CHECK(python_path_string("") == ".");
  CHECK(python_path_string("../x") == "../x");
}

TEST_CASE("real-data reduction tolerances apply only to named columns", "[csv][parity-policy]") {
  using guiding::table::parse_csv;
  using guiding::testing::compare_tables;
  const auto expected = parse_csv("iteration,energy_proxy,other,status\n1,1.0,1.0,ok\n");
  const std::map<std::string, double> tolerances{{"energy_proxy", 5e-15}};
  const auto compare = [&](const std::string& row) {
    return compare_tables(parse_csv("iteration,energy_proxy,other,status\n" + row + "\n"), expected, 0.0,
                          {}, tolerances);
  };

  CHECK(compare("1,1.000000000000004,1.0,ok").ok());
  CHECK_FALSE(compare("1,1.000000000000006,1.0,ok").ok());
  CHECK_FALSE(compare("1,1.0,1.0000000000000002,ok").ok());
  CHECK_FALSE(compare("2,1.0,1.0,ok").ok());
  CHECK_FALSE(compare("1,1.0,1.0,failed").ok());
  CHECK_FALSE(compare("1,,1.0,ok").ok());
  // The default remains exact when no per-column policy is supplied.
  CHECK_FALSE(compare_tables(parse_csv("iteration,energy_proxy,other,status\n1,1.000000000000004,1.0,ok\n"),
                             expected, 0.0).ok());
}

TEST_CASE("reduction tolerances preserve zero and nonfinite semantics", "[csv][parity-policy]") {
  using guiding::table::parse_csv;
  using guiding::testing::compare_tables;
  const auto compare = [](const std::string& actual, const std::string& expected) {
    return compare_tables(parse_csv("energy_proxy\n" + actual + "\n"),
                          parse_csv("energy_proxy\n" + expected + "\n"), 0.0, {}, {{"energy_proxy", 5e-15}});
  };
  CHECK(compare("0.0", "-0.0").ok());
  CHECK_FALSE(compare("1e-300", "0.0").ok());
  CHECK(compare("1.000000000000004e-200", "1e-200").ok());
  CHECK_FALSE(compare("1.00000000000001e-200", "1e-200").ok());
  CHECK(compare("nan", "nan").ok());
  CHECK_FALSE(compare("nan", "1.0").ok());
  CHECK_FALSE(compare("1.0", "nan").ok());
  CHECK_FALSE(compare("0.0", "nan").ok());
  CHECK_FALSE(compare("nan", "0.0").ok());
  CHECK(compare("inf", "inf").ok());
  CHECK(compare("-inf", "-inf").ok());
  CHECK_FALSE(compare("inf", "-inf").ok());
  CHECK_FALSE(compare("inf", "1.0").ok());
  CHECK_FALSE(compare("1.0", "inf").ok());
}
