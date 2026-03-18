cc_library(
    name = "mmcunet_models",
    srcs = [
        "models/mars_e300_comp_bgaf_watt_lstm.c",
        "models/mars_e300_comp_bgf_watt_lstm.c",
        "models/mars_e300_comp_gf_watt_lstm.c",
        "models/mars_e300_comp_gf_woatt_lstm.c",
        "models/mars_e300_comp_bgaf_watt_tcn.c",
        "models/mars_e300_comp_bgf_watt_tcn.c",
        "models/mars_e300_comp_gf_watt_tcn.c",
        "models/mars_e300_comp_gf_woatt_tcn.c",
        "models/mars_e300_comp_bgaf_watt_srnn.c",
        "models/mars_e300_comp_bgf_watt_srnn.c",
        "models/mars_e300_comp_gf_watt_srnn.c",
        "models/mars_e300_comp_gf_woatt_srnn.c",
    ],
    hdrs = [
        "models/mars_e300_comp_bgaf_watt_lstm.h",
        "models/mars_e300_comp_bgf_watt_lstm.h",
        "models/mars_e300_comp_gf_watt_lstm.h",
        "models/mars_e300_comp_gf_woatt_lstm.h",
        "models/mars_e300_comp_bgaf_watt_tcn.h",
        "models/mars_e300_comp_bgf_watt_tcn.h",
        "models/mars_e300_comp_gf_watt_tcn.h",
        "models/mars_e300_comp_gf_woatt_tcn.h",
        "models/mars_e300_comp_bgaf_watt_srnn.h",
        "models/mars_e300_comp_bgf_watt_srnn.h",
        "models/mars_e300_comp_gf_watt_srnn.h",
        "models/mars_e300_comp_gf_woatt_srnn.h",
    ],
)

cc_library(
    name = "macs_estimator",
    srcs = ["macs_estimator.cc"],
    hdrs = ["macs_estimator.h"],
    deps = [
        "//tensorflow/lite/schema:schema_utils",
    ],
)

cc_binary(
    name = "profile_mem_lat",
    srcs = ["profile_mem_lat.cc"],
    deps = [
        "//tensorflow/lite/schema:schema_utils",            # for schema_generated.h
        "//tensorflow/lite/micro:micro_framework",          # micro_interpreter.h
        "//tensorflow/lite/micro:micro_log",                # micro_log.h
        "//tensorflow/lite/micro:op_resolvers",             # micro_mutable_op_resolver.h
        "//tensorflow/lite/micro:micro_profiler",           # micro_profiler.h
        "//tensorflow/lite/micro:recording_allocators",     # recording_micro_interpreter.h
        "//tensorflow/lite/micro:system_setup",             # system_setup.h
        ":mmcunet_models",                                  # generated model data
        ":macs_estimator",
    ],
)
