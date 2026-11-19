# Copyright 2024 Sebastian Eibl
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

find_package(Doxygen REQUIRED OPTIONAL_COMPONENTS dot)

FetchContent_Declare(
        doxygen-awesome-css
        GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
        GIT_TAG 568f56cde6ac78b6dfcc14acd380b2e745c301ea #2.3.4
)
FetchContent_MakeAvailable(doxygen-awesome-css)
FetchContent_GetProperties(doxygen-awesome-css SOURCE_DIR AWESOME_CSS_DIR)

if (DOXYGEN_DOT_FOUND)
    set(MRMD_DOXYGEN_HAVE_DOT YES)
else ()
    set(MRMD_DOXYGEN_HAVE_DOT NO)
endif ()

# Doxygen >= 1.14 shows a page outline in a right-hand panel, but only on pages
# that have sections. Those pages get a narrower content column than the others,
# so turn it off. Older versions warn about the unknown tag, so set it only where
# it exists.
if (DOXYGEN_VERSION VERSION_GREATER_EQUAL 1.14)
    set(MRMD_DOXYGEN_VERSION_SPECIFIC "PAGE_OUTLINE_PANEL     = NO")
else ()
    set(MRMD_DOXYGEN_VERSION_SPECIFIC "")
endif ()

# docs/DoxygenLayout.xml only customizes the navigation tree. Doxygen does not
# reliably fall back to its defaults for the parts a layout file leaves out:
# 1.15 renders class, namespace and file pages empty. A full `doxygen -l` dump
# checked into the repository is no fix either, because it is tied to the
# version that wrote it. So take the default layout of the Doxygen in use and
# swap in our navindex element.
set(MRMD_DOXYGEN_LAYOUT ${CMAKE_CURRENT_BINARY_DIR}/DoxygenLayout.xml)
execute_process(
    COMMAND ${DOXYGEN_EXECUTABLE} -l DoxygenLayout.default.xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    RESULT_VARIABLE _layout_result
    OUTPUT_QUIET
)
if (NOT _layout_result EQUAL 0)
    message(FATAL_ERROR "`doxygen -l` failed with exit code ${_layout_result}")
endif ()
file(READ ${CMAKE_CURRENT_BINARY_DIR}/DoxygenLayout.default.xml _layout_default)
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/DoxygenLayout.xml _layout_custom)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/DoxygenLayout.xml)

# Cut out the navindex element with string(FIND): the custom block contains
# backslashes, which string(REGEX REPLACE) would interpret in the replacement.
function(mrmd_navindex_bounds text out_begin out_end)
    string(FIND "${text}" "<navindex>" _begin)
    string(FIND "${text}" "</navindex>" _end)
    if (_begin EQUAL -1 OR _end EQUAL -1)
        message(FATAL_ERROR "no <navindex> element found in a Doxygen layout")
    endif ()
    string(LENGTH "</navindex>" _close)
    math(EXPR _end "${_end} + ${_close}")
    set(${out_begin} ${_begin} PARENT_SCOPE)
    set(${out_end} ${_end} PARENT_SCOPE)
endfunction()

mrmd_navindex_bounds("${_layout_custom}" _cb _ce)
math(EXPR _clen "${_ce} - ${_cb}")
string(SUBSTRING "${_layout_custom}" ${_cb} ${_clen} _navindex)
mrmd_navindex_bounds("${_layout_default}" _db _de)
string(SUBSTRING "${_layout_default}" 0 ${_db} _head)
string(SUBSTRING "${_layout_default}" ${_de} -1 _tail)
file(WRITE ${MRMD_DOXYGEN_LAYOUT} "${_head}${_navindex}${_tail}")

configure_file(Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)

# The introductory prose for the generated tutorial overview page is passed as a
# file path, not as text: a multi-line argument does not survive the generated
# build rules.
#
# --strict fails the build if a tutorial lost its "//> <Title>" line, which would
# otherwise silently publish a page titled with a sentence fragment.
add_custom_target(tutorial_index
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/gen_tutorial_index.py
          ${CMAKE_SOURCE_DIR}/examples ${CMAKE_BINARY_DIR}/docs/tutorials.dox
          --recursive --strict
          --intro-file ${CMAKE_CURRENT_SOURCE_DIR}/tutorials_intro.txt
    VERBATIM)

add_custom_target(docs
        COMMAND Doxygen::doxygen ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
add_dependencies(docs tutorial_index)

