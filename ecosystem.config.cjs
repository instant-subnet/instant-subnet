module.exports = {
  apps: [
    {
      name: "instant-validator",
      cwd: __dirname,
      script: ".venv/bin/python",
      args: "-m instant_validator",
      interpreter: "none",
      autorestart: true,
      restart_delay: 5000,
      max_restarts: 10,
      min_uptime: "30s",
      kill_timeout: 60000,
      time: true,
      env: {
        PYTHONUNBUFFERED: "1",
      },
    },
  ],
};
