"""Patch ServeurpersoCom/omnivoice.cpp tts-server to support JSON 'steps', 'denoise', and --api-key."""

from __future__ import annotations

import argparse
from pathlib import Path

PINNED_OMNIVOICE_COMMIT = "040c8b344d8c670ce1475194751d119b5ef82c78"


def patch_omnivoice_sources(repo_dir: Path) -> None:
    """Idempotently patch src/tts-server.h and tools/tts-server.cpp."""
    h_path = repo_dir / "src" / "tts-server.h"
    cpp_path = repo_dir / "tools" / "tts-server.cpp"
    if not h_path.is_file() or not cpp_path.is_file():
        raise FileNotFoundError(f"Expected omnivoice.cpp sources in {repo_dir}")

    # ------------------------------------------------------------------
    # 1. Patch src/tts-server.h
    # ------------------------------------------------------------------
    h_src = h_path.read_text(encoding="utf-8")

    if "int     steps;" not in h_src:
        old_req = "    int64_t seed;\n};"
        new_req = (
            "    int64_t seed;\n"
            "    int     steps;\n"
            "    int     denoise; // -1 = default (true), 0 = false, 1 = true\n"
            "};"
        )
        if old_req not in h_src:
            raise RuntimeError("Anchor not found in src/tts-server.h: struct tts_request")
        h_src = h_src.replace(old_req, new_req, 1)

    if "std::string api_key;" not in h_src:
        old_cfg = "    int         port = 8080;\n};"
        new_cfg = "    int         port = 8080;\n    std::string api_key;\n};"
        if old_cfg not in h_src:
            raise RuntimeError("Anchor not found in src/tts-server.h: struct server_config")
        h_src = h_src.replace(old_cfg, new_cfg, 1)

    if 'yyjson_obj_get(root, "steps")' not in h_src:
        old_parse = "    req.seed = seed ? yyjson_get_sint(seed) : -1;"
        new_parse = (
            "    req.seed = seed ? yyjson_get_sint(seed) : -1;\n\n"
            '    yyjson_val * steps_v = yyjson_obj_get(root, "steps");\n'
            "    if (steps_v && !yyjson_is_int(steps_v)) {\n"
            '        err = "\'steps\' must be an integer";\n'
            "        yyjson_doc_free(doc);\n"
            "        return false;\n"
            "    }\n"
            "    req.steps = steps_v ? (int) yyjson_get_sint(steps_v) : 0;\n"
            "    if (req.steps < 0 || req.steps > 128) {\n"
            '        err = "\'steps\' must be between 1 and 128";\n'
            "        yyjson_doc_free(doc);\n"
            "        return false;\n"
            "    }\n\n"
            '    yyjson_val * denoise_v = yyjson_obj_get(root, "denoise");\n'
            "    if (denoise_v && !yyjson_is_bool(denoise_v)) {\n"
            '        err = "\'denoise\' must be a boolean";\n'
            "        yyjson_doc_free(doc);\n"
            "        return false;\n"
            "    }\n"
            "    req.denoise = denoise_v ? (yyjson_get_bool(denoise_v) ? 1 : 0) : -1;"
        )
        if old_parse not in h_src:
            raise RuntimeError("Anchor not found in src/tts-server.h: seed parsing")
        h_src = h_src.replace(old_parse, new_parse, 1)

    if "set_pre_routing_handler" not in h_src:
        old_cors = (
            "    // permissive CORS so a browser client can call the API directly.\n"
            "    svr.set_default_headers({\n"
            '        { "Access-Control-Allow-Origin", "*" }\n'
            "    });\n"
            '    svr.Options("/.*", [](const httplib::Request &, httplib::Response & res) {\n'
            '        res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");\n'
            '        res.set_header("Access-Control-Allow-Headers", "Content-Type");\n'
            "    });"
        )
        new_cors = (
            "    if (cfg.api_key.empty()) {\n"
            "        // permissive CORS only when running without an API key.\n"
            "        svr.set_default_headers({\n"
            '            { "Access-Control-Allow-Origin", "*" }\n'
            "        });\n"
            '        svr.Options("/.*", [](const httplib::Request &, httplib::Response & res) {\n'
            '            res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");\n'
            '            res.set_header("Access-Control-Allow-Headers", "Content-Type");\n'
            "        });\n"
            "    } else {\n"
            "        const std::string expected_key = cfg.api_key;\n"
            "        const std::string expected_bearer = \"Bearer \" + cfg.api_key;\n"
            "        svr.set_pre_routing_handler(\n"
            "            [expected_key, expected_bearer](const httplib::Request & req, httplib::Response & res) {\n"
            '                if (req.path == "/health") {\n'
            "                    return httplib::Server::HandlerResponse::Unhandled;\n"
            "                }\n"
            '                const std::string hdr_key = req.get_header_value("X-Api-Key");\n'
            '                const std::string hdr_auth = req.get_header_value("Authorization");\n'
            "                if (hdr_key == expected_key || hdr_auth == expected_bearer) {\n"
            "                    return httplib::Server::HandlerResponse::Unhandled;\n"
            "                }\n"
            '                tts_json_error(res, 401, "unauthorized", "missing or invalid API key");\n'
            "                return httplib::Server::HandlerResponse::Handled;\n"
            "            });\n"
            "    }"
        )
        if old_cors not in h_src:
            raise RuntimeError("Anchor not found in src/tts-server.h: CORS block")
        h_src = h_src.replace(old_cors, new_cors, 1)

    h_path.write_text(h_src, encoding="utf-8")

    # ------------------------------------------------------------------
    # 2. Patch tools/tts-server.cpp
    # ------------------------------------------------------------------
    cpp_src = cpp_path.read_text(encoding="utf-8")

    if "supports 'steps'" not in cpp_src:
        old_usage = '            "  --clamp-fp16            Clamp hidden states to FP16 range\\n",'
        new_usage = (
            '            "  --clamp-fp16            Clamp hidden states to FP16 range\\n"\n'
            '            "  --api-key <token>       Require X-Api-Key header and disable wildcard CORS\\n"\n'
            '            "  JSON /v1/audio/speech:  supports \'steps\' (MaskGIT decode steps) and \'denoise\'\\n",'
        )
        if old_usage not in cpp_src:
            raise RuntimeError("Anchor not found in tools/tts-server.cpp: print_usage")
        cpp_src = cpp_src.replace(old_usage, new_usage, 1)

    if '"--api-key"' not in cpp_src:
        old_arg = '        } else if (!std::strcmp(arg, "--clamp-fp16")) {\n            clamp_fp16 = true;'
        new_arg = (
            '        } else if (!std::strcmp(arg, "--clamp-fp16")) {\n'
            "            clamp_fp16 = true;\n"
            '        } else if (!std::strcmp(arg, "--api-key") && i + 1 < argc) {\n'
            "            cfg.api_key = argv[++i];"
        )
        if old_arg not in cpp_src:
            raise RuntimeError("Anchor not found in tools/tts-server.cpp: CLI args")
        cpp_src = cpp_src.replace(old_arg, new_arg, 1)

    if "p.mg_num_step = req.steps;" not in cpp_src:
        old_seed = "        p.mg_seed = (req.seed < 0) ? (uint64_t) std::random_device{}() : (uint64_t) req.seed;"
        new_seed = (
            old_seed + "\n"
            "        if (req.steps > 0) {\n"
            "            p.mg_num_step = req.steps;\n"
            "        }\n"
            "        if (req.denoise >= 0) {\n"
            "            p.denoise = (req.denoise != 0);\n"
            "        }"
        )
        if old_seed not in cpp_src:
            raise RuntimeError("Anchor not found in tools/tts-server.cpp: p.mg_seed")
        cpp_src = cpp_src.replace(old_seed, new_seed, 1)

    cpp_path.write_text(cpp_src, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch omnivoice.cpp tts-server sources.")
    parser.add_argument("repo_dir", type=Path, help="Path to cloned omnivoice.cpp repository")
    args = parser.parse_args()
    patch_omnivoice_sources(args.repo_dir)
    print(f"Patched omnivoice.cpp tts-server successfully in {args.repo_dir}")


if __name__ == "__main__":
    main()
