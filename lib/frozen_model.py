"""Where the frozen model's runs are. B0 = the configuration frozen in PAPER_3 §7 (5 folds x seeds 42, 43).

B0_RUN           runs/sweep4/B0: B0 retrained on 2026-10-16 with the Carrington rotation rate (config.Omega = 2.8653e-6);
                 the model of the papers.
B0_RUN_OMEGA297  runs/sweep3/B0: the earlier training with Omega = 2.97e-6 (24.49 d, 3.6 % too fast); kept for reference
                 and for the regression check, not to be reported.
"""
B0_RUN = 'runs/sweep4/B0'
B0_RUN_OMEGA297 = 'runs/sweep3/B0'
