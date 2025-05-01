# Vision AI

## Overview

Vision AI is a web application designed to streamline the clip-finding workflow for video editors who work with large local video libraries. By ingesting and segmenting raw footage into meaningful “chunks,” embedding them into a vector database, and generating concise summaries, Vision AI provides an efficient, AI-powered search interface for locating B-roll, sound bites, or any moment in a video—saving editors hours of manual searching and playback.

## Problem

Video editors often juggle hundreds of gigabytes of raw footage stored haphazardly across hard drives or network shares. Finding the right clip for a YouTuber’s next video—whether it’s a reaction shot, establishing B-roll, or a specific sound bite—can be time-consuming and error-prone, even for the most organized workflows. Editors need to:

- Locate specific scenes or subjects without re-watching entire files or going through a full repository of folders.  
- Search across multiple folders and storage devices in one go  
- Retrieve contextual information (e.g., who’s on screen, what’s happening, when it occurs)

## User Profiles

- **Freelance Video Editors**  
  - Need fast access to clips across many projects  
  - Often switch between clients and storage locations  
- **In-House Production Teams**  
  - Collaborate on large volumes of footage  
  - Require shared, searchable archives  
- **Content Creators / YouTubers**  
  - May not have an organized folder structure  
  - Want to empower their editors with better tooling  

## Features

1. **AI-Powered Clip Segmentation & Summarization**  
   - Automatically split uploaded videos into semantically coherent segments  
   - Generate short text summaries and key-frame thumbnails for each segment  

2. **Vector Embedding & Search**  
   - Embed summaries and thumbnails into ChromaDB (vector database)  
   - Keyword, semantic, and example-based search to surface relevant segments  

3. **Intuitive Web UI**  
   - Global “Search Clips” bar with autocomplete and filter scopes (date, project, speaker)  
   - Results list showing thumbnail, summary snippet, and timecode  
   - “Play in Context” preview player for 5–10 second clip playback  

4. **Project & Folder Management**  
   - Organize videos into projects or collections  
   - Bulk-upload via drag & drop or folder syncing  

5. **User Accounts & Permissions**  
   - Sign up / log in via email & password  
   - Team invites & role-based access (Viewer, Editor, Admin)  

## Tech Stack

- **Frontend**: React, TypeScript, Tailwind CSS, React Router, Axios  
- **Backend**: Node.js, Express, Agentic LLM Workflow (open-source models + RLHF)  
- **Database**:  
  - **Vector DB**: ChromaDB for embeddings & semantic search  
  - **Relational DB**: MySQL or PostgreSQL for user data & metadata  
- **AI / ML**:  
  - Open-source LLMs fine-tuned on video-caption datasets (or prompt-based few-shot pipelines)  
  - Custom agent orchestration layer for video ingestion & summarization  

## Architecture & Data Flow

1. **Upload / Ingest**  
   - User uploads video → stored on CDN or object storage  
2. **Segmentation Service**  
   - Background worker splits video into N-second windows.
3. **Summarization & Embedding**  
   - Each segment passed to an LLM agent for summary + key-frame extraction  
   - Summaries embedded via sentence transformers → vectors stored in ChromaDB  
4. **Search API**  
   - Frontend issues semantic or keyword queries → hits ChromaDB → returns segment IDs  
   - Metadata & timecodes fetched from relational DB → assembled into results 