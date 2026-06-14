"""
src — AI4I 2020 Predictive Maintenance ML Pipeline
====================================================

A modular, production-grade pipeline for multi-class predictive
maintenance on the AI4I 2020 dataset.

Modules
-------
data_loader          Load, validate, and prepare the raw CSV dataset.
feature_engineering   Create domain-driven engineered features.
preprocessing        Build sklearn column transformers and sampling strategies.
train                Full ablation study across feature sets × samplers × models.
evaluate             Compute metrics, quality gates, and generate reports.
predict              Load a trained model and run inference.
"""

__version__ = "1.0.0"
