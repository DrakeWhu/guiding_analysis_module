#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "guiding/products/singlecase_score.hpp"
#include "guiding/table/csv.hpp"
#include "guiding/table/record.hpp"

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
  using guiding::products::python_path_string;
  CHECK(python_path_string("./out//case/") == "out/case");
  CHECK(python_path_string("/a/./b/") == "/a/b");
  CHECK(python_path_string("") == ".");
  CHECK(python_path_string("../x") == "../x");
}
