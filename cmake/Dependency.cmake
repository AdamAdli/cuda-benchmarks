#
# Functions Exported
#   - download_dependency
#   - find_dependency
#   - find_or_download_dependency
#

include(ExternalProject)

#
#   Environment
#

IF(NOT DEFINED DOWNLOAD_DEPENDENCY_GIT_USE_SSH)
    set(DOWNLOAD_DEPENDENCY_GIT_USE_SSH TRUE)
ENDIF()

#
#   Library Function
#

function(download_dependency)
    set(options
            HEADER_ONLY
            SKIP_INSTALL
            SKIP_CONFIGURE
            )
    set(oneValueArgs
            NAME
            BUILT_LIB_NAME # Only used with SKIP_INSTALL
            BUILT_LIB_PATH
            GIT_REPOSITORY
            GIT_REPOSITORY_SSH
            GIT_TAG
            URL
            CMAKE_BUILD_TYPE
            LIB_PATH_SUFFIX # Only used with install
                            # TODO: Figure out naming consistency with BUILT_LIB_NAME and BUILT_LIB_PATH
            )
    set(multiValueArgs
            ADDITIONAL_LIBS
            ADDITIONAL_INCLUDE_DIRS
            ADDITIONAL_CMAKE_ARGS
            DEPENDS
            )
    cmake_parse_arguments(PARSE_ARGV 0 DOWNLOAD_DEPENDENCY "${options}" "${oneValueArgs}" "${multiValueArgs}")

    set(NAME ${DOWNLOAD_DEPENDENCY_NAME})
    set(GIT_REPOSITORY ${DOWNLOAD_DEPENDENCY_GIT_REPOSITORY})
    set(GIT_TAG ${DOWNLOAD_DEPENDENCY_GIT_TAG})
    set(HEADER_ONLY ${DOWNLOAD_DEPENDENCY_HEADER_ONLY})
    set(SKIP_INSTALL ${DOWNLOAD_DEPENDENCY_SKIP_INSTALL})
    set(SKIP_CONFIGURE ${DOWNLOAD_DEPENDENCY_SKIP_CONFIGURE})
    set(_CMAKE_BUILD_TYPE ${DOWNLOAD_DEPENDENCY_CMAKE_BUILD_TYPE}) # if not set mirrors global CMAKE_BUILD_TYPE
    set(ADDITIONAL_LIBS ${DOWNLOAD_DEPENDENCY_ADDITIONAL_LIBS})
    set(ADDITIONAL_INCLUDE_DIRS ${DOWNLOAD_DEPENDENCY_ADDITIONAL_INCLUDE_DIRS})
    set(ADDITIONAL_CMAKE_ARGS ${DOWNLOAD_DEPENDENCY_ADDITIONAL_CMAKE_ARGS})
    set(DEPENDS ${DOWNLOAD_DEPENDENCY_DEPENDS})
    set(LIB_PATH_SUFFIX  ${DOWNLOAD_DEPENDENCY_LIB_PATH_SUFFIX})

    IF(${DOWNLOAD_DEPENDENCY_GIT_USE_SSH} AND DEFINED DOWNLOAD_DEPENDENCY_GIT_REPOSITORY_SSH)
        set(GIT_REPOSITORY ${DOWNLOAD_DEPENDENCY_GIT_REPOSITORY_SSH})
        message(STATUS "[${NAME}] Using ssh: ${GIT_REPOSITORY}")
    ENDIF()

    IF(DEFINED DOWNLOAD_DEPENDENCY_URL)
        set(DOWNLOAD_INFO URL ${DOWNLOAD_DEPENDENCY_URL})
    ELSE()
        set(DOWNLOAD_INFO GIT_REPOSITORY ${GIT_REPOSITORY} GIT_TAG ${GIT_TAG})
    ENDIF()

    IF(DEFINED ADDITIONAL_LIBS)
        set(ADDITIONAL_LIBS_SET TRUE)
    ELSE()
        set(ADDITIONAL_LIBS_SET FALSE)
    ENDIF()

    IF(DEFINED ADDITIONAL_INCLUDE_DIRS)
        set(ADDITIONAL_INCLUDE_DIRS_SET TRUE)
    ELSE()
        set(ADDITIONAL_INCLUDE_DIRS_SET FALSE)
    ENDIF()

    IF(DEFINED DOWNLOAD_DEPENDENCY_BUILT_LIB_PATH)
        set(BUILT_LIB_PATH ${DOWNLOAD_DEPENDENCY_BUILT_LIB_PATH})
        set(BUILT_LIB_PATH_SET TRUE)
    ELSE()
        set(BUILT_LIB_PATH_SET FALSE)
    ENDIF()

    IF(DEFINED DOWNLOAD_DEPENDENCY_BUILT_LIB_NAME)
        set(BUILT_LIB_NAME ${DOWNLOAD_DEPENDENCY_BUILT_LIB_NAME})
    ELSE()
        string(TOLOWER ${NAME} BUILT_LIB_NAME)
    ENDIF()

    IF(${HEADER_ONLY})
        set(REQUIRED_STEP configure)
    ELSEIF(${SKIP_INSTALL})
        set(REQUIRED_STEP build)
    ELSE()
        set(REQUIRED_STEP install)
    ENDIF()

    message(STATUS "[${NAME}] Set to download and ${REQUIRED_STEP} on build")

    IF(DEFINED _CMAKE_BUILD_TYPE)
        set(CMAKE_ARGS "-DCMAKE_BUILD_TYPE=${_CMAKE_BUILD_TYPE}")
    ELSE()
        set(CMAKE_ARGS "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
    ENDIF()

    IF(NOT DEFINED LIB_PATH_SUFFIX)
        set(LIB_PATH_SUFFIX lib)
    ENDIF()

    set(INSTALL_DIR "")

    IF(${REQUIRED_STEP} STREQUAL "install")
        set(INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}/external/installed/${NAME}_download)
        list(APPEND CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR})
    ENDIF()

    list(APPEND CMAKE_ARGS ${ADDITIONAL_CMAKE_ARGS})
    set(SOURCE_DIR  ${CMAKE_CURRENT_BINARY_DIR}/external/src/${NAME}_download)

    ExternalProject_Add(${NAME}_download
        PREFIX            external
        ${DOWNLOAD_INFO} # currently either (GIT_REPOSITORY and GIT_TAG) or URL
        STEP_TARGETS      ${REQUIRED_STEP}
        EXCLUDE_FROM_ALL  TRUE
        UPDATE_COMMAND    ""
        INSTALL_DIR       ${INSTALL_DIR}
        CMAKE_ARGS        ${CMAKE_ARGS}
        DEPENDS           ${DEPENDS}
    )

    cmake_language(EVAL CODE "
        macro(add_dependency_${NAME} TARGET)
            #
            #   Add as depenedency if STEPs (i.e. configure, build, install) are required
            #
            IF(NOT ${HEADER_ONLY} OR NOT ${SKIP_CONFIGURE})
                add_dependencies(\${TARGET} ${NAME}_download-${REQUIRED_STEP})
            ENDIF()

            #   If a build step is going to run log the CMAKE_ARGS used
            IF(\"${REQUIRED_STEP}\" STREQUAL \"install\" OR \"${REQUIRED_STEP}\" STREQUAL \"build\")
                message(STATUS \"[\${TARGET}][${NAME}] Using cmake args: ${CMAKE_ARGS}\")
            ENDIF()

            #
            # Add include directories
            #
            IF(\"${REQUIRED_STEP}\" STREQUAL \"install\")
                target_include_directories(\${TARGET} PUBLIC ${INSTALL_DIR}/include)
                message(STATUS \"[\${TARGET}][${NAME}] Adding include dir: ${INSTALL_DIR}/include\")
            ELSE()
                # Nothing was installed get the include direcotories from the source
                #   this is for HEADER_ONLY or SKIP_INSTALL mode
                target_include_directories(\${TARGET} PUBLIC ${SOURCE_DIR})
                target_include_directories(\${TARGET} PUBLIC ${SOURCE_DIR}/include)
                message(STATUS \"[\${TARGET}][${NAME}] Adding include dir: ${SOURCE_DIR}\")
                message(STATUS \"[\${TARGET}][${NAME}] Adding include dir: ${SOURCE_DIR}/include\")
            ENDIF()

            IF(${ADDITIONAL_INCLUDE_DIRS_SET})
                FOREACH(ADDITIONAL_INCLUDE_DIR IN ITEMS ${ADDITIONAL_INCLUDE_DIRS})
                    message(STATUS \"[\${TARGET}][${NAME}] Adding include dir: ${SOURCE_DIR}/\${ADDITIONAL_INCLUDE_DIR}\")
                    target_include_directories(\${TARGET} PUBLIC ${SOURCE_DIR}/\${ADDITIONAL_INCLUDE_DIR})
                ENDFOREACH()
            ENDIF()

            #
            # Add link directories
            #
            IF(\"${REQUIRED_STEP}\" STREQUAL \"install\")
                target_link_directories(\${TARGET} PUBLIC ${INSTALL_DIR}/${LIB_PATH_SUFFIX})
            ENDIF()
            IF (${BUILT_LIB_PATH_SET})
                target_link_directories(\${TARGET} PUBLIC ${SOURCE_DIR}-build/${BUILT_LIB_PATH})
            ENDIF()

            #
            # Link libraries
            #
            IF (NOT ${HEADER_ONLY})
                target_link_libraries(\${TARGET} ${BUILT_LIB_NAME})
                message(STATUS \"[\${TARGET}][${NAME}] Linking lib: ${BUILT_LIB_NAME}\")
                IF(${ADDITIONAL_LIBS_SET})
                    target_link_libraries(\${TARGET} ${ADDITIONAL_LIBS})
                    message(STATUS \"[\${TARGET}][${NAME}] Linking additional libs: ${ADDITIONAL_LIBS}\")
                ENDIF()
            ENDIF()
        endmacro()"
    )
endfunction()

function(find_dependency)
    set(options REQUIRED HEADER_ONLY)
    set(oneValueArgs NAME PREFIX_NAME)
    set(multipleValueArgs EXPORT)
    # NOTE: Also need to supply the args for `download_dependency`
    cmake_parse_arguments(PARSE_ARGV 0 FIND_DEPENDENCY "${options}" "${oneValueArgs}" "${multipleValueArgs}")

    set(NAME ${FIND_DEPENDENCY_NAME})
    set(HEADER_ONLY ${FIND_DEPENDENCY_HEADER_ONLY})
    set(EXPORT ${FIND_DEPENDENCY_EXPORT})

    IF(DEFINED FIND_DEPENDENCY_PREFIX_NAME)
        set(PREFIX_NAME ${FIND_DEPENDENCY_PREFIX_NAME})
    ELSE()
        set(PREFIX_NAME ${NAME})
    ENDIF()

    IF(${FIND_DEPENDENCY_REQUIRED})
        set(FIND_PACKAGE_OPTIONS REQUIRED)
    ELSE()
        set(FIND_PACKAGE_OPTIONS QUIET)
    ENDIF()

    find_package(${NAME} ${FIND_PACKAGE_OPTIONS})

    FOREACH(VAR_TO_EXPORT IN ITEMS ${EXPORT})
        cmake_language(EVAL CODE "set(${VAR_TO_EXPORT} \"\${${VAR_TO_EXPORT}}\" PARENT_SCOPE)")
    ENDFOREACH()

    IF(NOT DEFINED ${PREFIX_NAME}_VERSION)
        set(${PREFIX_NAME}_VERSION "unknown")
    ENDIF()

    IF(${PREFIX_NAME}_FOUND)
        message(STATUS "[${NAME}] Found version: \"${${PREFIX_NAME}_VERSION}\" on the machine, using that")
        set(${PREFIX_NAME}_FOUND TRUE PARENT_SCOPE)
        cmake_language(EVAL CODE "
            macro(add_dependency_${NAME} TARGET)

                IF(NOT \"${${PREFIX_NAME}_INCLUDE_DIR}\" STREQUAL \"\")
                    message(STATUS \"[\${TARGET}][${NAME}] Adding include dir: ${${PREFIX_NAME}_INCLUDE_DIR}\")
                    target_include_directories(\${TARGET} PUBLIC ${${PREFIX_NAME}_INCLUDE_DIR})
                ENDIF()

                IF(NOT \"${${PREFIX_NAME}_INCLUDE_DIRS}\" STREQUAL \"\")
                    message(STATUS \"[\${TARGET}][${NAME}] Adding include dirs: ${${PREFIX_NAME}_INCLUDE_DIRS}\")
                    target_include_directories(\${TARGET} PUBLIC ${${PREFIX_NAME}_INCLUDE_DIRS})
                ENDIF()

                # TODO: Clean up, for now we have to use `STREQUAL \"\"` instead of `DEFINED` since the libraries won't
                #   be available when the macro is run so we need to expand them now
                IF(NOT ${HEADER_ONLY} AND
                    (NOT \"${${PREFIX_NAME}_LIBRARY}\" STREQUAL \"\"
                     OR NOT \"${${PREFIX_NAME}_LIBRARIES}\" STREQUAL \"\"))

                    message(STATUS
                        \"[\${TARGET}][${NAME}] Linking lib: ${${PREFIX_NAME}_LIBRARY} ${${PREFIX_NAME}_LIBRARIES}\")
                    target_link_libraries(\${TARGET}
                        ${${PREFIX_NAME}_LIBRARY}
                        ${${PREFIX_NAME}_LIBRARIES})
                ENDIF()
            endmacro()"
        )
    ENDIF()
endfunction()

function(find_or_download_dependency)
    set(options HEADER_ONLY)
    set(oneValueArgs NAME PREFIX_NAME ON_FAILED_TO_FIND)
    # NOTE: Also need to supply the args for `download_dependency`
    cmake_parse_arguments(PARSE_ARGV 0 FIND_OR_DOWNLOAD_DEPENDENCY "${options}" "${oneValueArgs}" "")

    set(NAME ${FIND_OR_DOWNLOAD_DEPENDENCY_NAME})
    set(HEADER_ONLY ${FIND_OR_DOWNLOAD_DEPENDENCY_HEADER_ONLY})
    set(ON_FAILED_TO_FIND ${FIND_OR_DOWNLOAD_DEPENDENCY_ON_FAILED_TO_FIND})

    IF(DEFINED FIND_OR_DOWNLOAD_DEPENDENCY_PREFIX_NAME)
        set(PREFIX_NAME ${FIND_OR_DOWNLOAD_DEPENDENCY_PREFIX_NAME})
    ELSE()
        set(PREFIX_NAME ${NAME})
    ENDIF()

    set(FORCE_DOWNLOAD FALSE)
    IF(DEFINED DEPENDENCY_FORCE_DOWNLOAD)
        list (FIND DEPENDENCY_FORCE_DOWNLOAD ${NAME} _index)
        IF(${_index} GREATER -1)
            set(FORCE_DOWNLOAD TRUE)
            message(STATUS "[${NAME}] Force downloading")
        ENDIF()
    ENDIF()

    IF(NOT ${FORCE_DOWNLOAD})
        find_dependency(${ARGV})
        set(${PREFIX_NAME}_FOUND ${PREFIX_NAME}_FOUND PARENT_SCOPE) # Roll-up from find_dependency to caller
    ENDIF()

    IF(NOT ${PREFIX_NAME}_FOUND)
        IF(DEFINED ON_FAILED_TO_FIND)
            cmake_language(EVAL CODE "${ON_FAILED_TO_FIND}()")
        ENDIF()

        IF(NOT ${FORCE_DOWNLOAD})
            message(STATUS "[${NAME}] Failed to find on machine")
        ENDIF()
        download_dependency(${ARGV})
    ENDIF()
endfunction()
