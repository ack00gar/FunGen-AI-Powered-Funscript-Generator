import subprocess


def test_cli_with_all_params():
    video_path = "C:/cvr/funscript-generator/test_koogar_extra_short.mp4"

    command = [
        "python", "-m", "script_generator.cli.generate_funscript",
        video_path,
        "--reuse-yolo", "True",
        "--copy-funscript", "True"
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    print(result.stdout)
    print(result.stderr)

if __name__ == "__main__":
    test_cli_with_all_params()