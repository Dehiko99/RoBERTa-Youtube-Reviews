# 🎬 YouTube Comments Analysis — Sentiment & Topic Modeling

End-to-end NLP pipeline for analyzing YouTube comments on movies using RoBERTa for sentiment analysis and BERTopic for topic modeling, with full database integration.

📌 Overview
This project builds a complete pipeline that:

1. Collects YouTube comments for selected movies through HTTP scraping via requests, no API key required
2. Stores raw and processed data in a structured database
3. Analyzes sentiment at comment level using a pre-trained RoBERTa model
4. Discovers latent topics across the comment corpus using BERTopic
5. Produces insights on how audiences discuss and react to specific films

The pipeline is fully automated — from data ingestion to final output — and operates end-to-end with persistent storage at each stage.

