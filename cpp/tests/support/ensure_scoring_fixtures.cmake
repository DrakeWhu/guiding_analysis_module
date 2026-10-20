# CSV-only scoring cases are discovered through their otherwise empty campaign
# directories. The manifest includes the deliberately missing-products case,
# so inferring the hierarchy from case_metrics would omit part of the fixture.
if(NOT DEFINED GUIDING_SCORING_FIXTURES_DIR)
  set(GUIDING_SCORING_FIXTURES_DIR "${CMAKE_CURRENT_LIST_DIR}/../data/synthetic/scoring")
endif()
file(STRINGS "${GUIDING_SCORING_FIXTURES_DIR}/campaign_cases.txt" scoring_cases)
if(NOT scoring_cases)
  message(FATAL_ERROR "The scoring campaign fixture manifest is empty")
endif()
foreach(case_name IN LISTS scoring_cases)
  file(MAKE_DIRECTORY "${GUIDING_SCORING_FIXTURES_DIR}/campaign/${case_name}")
endforeach()
