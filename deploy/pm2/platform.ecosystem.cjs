const path = require("node:path");

const root = path.resolve(__dirname, "../..");

module.exports = {
  apps: [
    {
      name: "instant-platform",
      cwd: root,
      script: path.join(root, ".venv/bin/python"),
      args: ["-m", "instant.platform"],
      interpreter: "none",
      exec_mode: "fork",
      instances: 1,
      autorestart: true,
      restart_delay: 5000,
      exp_backoff_restart_delay: 100,
      kill_timeout: 30000,
      max_memory_restart: "1G",
      time: true,
      out_file: path.join(root, "logs/platform.out.log"),
      error_file: path.join(root, "logs/platform.error.log"),
      merge_logs: true,
      env: {
        PYTHONUNBUFFERED: "1",
        INSTANT_CONFIG_DIR:
          process.env.INSTANT_CONFIG_DIR || path.join(root, "config"),
      },
    },
  ],
};
