---
title: SAVI Valuation Agent
emoji: 🏠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Real-estate valuation agent for Ames, Iowa (XGBoost + RL)
---

# SAVI — autonomous real-estate valuation agent (Ames, Iowa)

XGBoost valuation model (out-of-fold R²(log) 0.921, MAPE 7.8%) plus a reinforcement-learning
policy that approves, sends to review or rejects each automatic valuation. The chat agent answers
only with figures returned by its tools.

Source, methodology and evaluation: https://github.com/Julian-Rincon/ames-housing-ml
