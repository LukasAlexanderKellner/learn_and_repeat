SHELL := /bin/bash

PYTHON ?= .venv/bin/python
PYTHON := $(abspath $(PYTHON))
COMPOSE = docker compose --env-file docker/.env -f docker/docker-compose.yml
PID_DIR := .run
WEBUI_PID_FILE := $(PID_DIR)/webui.pid
WEBUI_LOG_FILE := $(PID_DIR)/webui.log

APP_PORT ?= $(shell awk -F= '$$1 == "APP_PORT" { print $$2 }' docker/.env 2>/dev/null)
ifeq ($(strip $(APP_PORT)),)
APP_PORT := 8080
endif

.PHONY: help start_app stop_app start_db stop_db start_webui stop_webui ingest

help:
	@printf '%s\n' \
	  'start_app   Start database, then NiceGUI web UI' \
	  'stop_app    Stop NiceGUI web UI, then database container' \
	  'start_db    Start PostgreSQL container' \
	  'stop_db     Stop PostgreSQL container without deleting data' \
	  'start_webui Start NiceGUI web UI in the background' \
	  'stop_webui  Stop NiceGUI web UI' \
	  'ingest      OCR unprocessed PDFs, then chunk, embed, and import Markdown'

start_app:
	$(MAKE) start_db
	$(MAKE) start_webui

stop_app:
	$(MAKE) stop_webui
	$(MAKE) stop_db

start_db:
	test -f docker/.env || { echo "Missing docker/.env; copy docker/.env.example and configure it."; exit 1; }
	$(COMPOSE) up -d --wait postres

stop_db:
	$(COMPOSE) stop postgres

start_webui:
	@set -euo pipefail; \
	mkdir -p "$(PID_DIR)"; \
	if [[ -f "$(WEBUI_PID_FILE)" ]]; then \
	  pid="$$(cat "$(WEBUI_PID_FILE)")"; \
	  if kill -0 "$$pid" 2>/dev/null && ps -p "$$pid" -o command= | grep -Fq 'src.app.web_ui'; then \
	    echo "NiceGUI is already running (PID $$pid)."; \
	    exit 0; \
	  fi; \
	  rm -f "$(WEBUI_PID_FILE)"; \
	fi; \
	APP_PORT="$(APP_PORT)" nohup "$(PYTHON)" -m src.app.web_ui >>"$(WEBUI_LOG_FILE)" 2>&1 </dev/null & \
	pid=$$!; \
	echo "$$pid" >"$(WEBUI_PID_FILE)"; \
	for attempt in {1..20}; do \
	  if ! kill -0 "$$pid" 2>/dev/null; then \
	    echo "NiceGUI failed to start. Recent log output:"; \
	    tail -n 40 "$(WEBUI_LOG_FILE)"; \
	    rm -f "$(WEBUI_PID_FILE)"; \
	    exit 1; \
	  fi; \
	  if curl --silent --fail "http://127.0.0.1:$(APP_PORT)/" >/dev/null; then \
	    echo "NiceGUI is running at http://127.0.0.1:$(APP_PORT)/ (PID $$pid)."; \
	    exit 0; \
	  fi; \
	  sleep 0.5; \
	done; \
	echo "NiceGUI did not become responsive. Recent log output:"; \
	tail -n 40 "$(WEBUI_LOG_FILE)"; \
	exit 1

stop_webui:
	@set -euo pipefail; \
	if [[ ! -f "$(WEBUI_PID_FILE)" ]]; then \
	  echo "No NiceGUI PID file found; the web UI may not be running."; \
	  exit 0; \
	fi; \
	pid="$$(cat "$(WEBUI_PID_FILE)")"; \
	if ! kill -0 "$$pid" 2>/dev/null; then \
	  rm -f "$(WEBUI_PID_FILE)"; \
	  echo "Removed stale NiceGUI PID file."; \
	  exit 0; \
	fi; \
	if ! ps -p "$$pid" -o command= | grep -Fq 'src.app.web_ui'; then \
	  echo "PID $$pid is not the NiceGUI process; refusing to stop it."; \
	  exit 1; \
	fi; \
	kill -TERM "$$pid"; \
	for attempt in {1..20}; do \
	  if ! kill -0 "$$pid" 2>/dev/null; then \
	    rm -f "$(WEBUI_PID_FILE)"; \
	    echo "NiceGUI stopped."; \
	    exit 0; \
	  fi; \
	  sleep 0.5; \
	done; \
	echo "NiceGUI did not stop after SIGTERM; leaving the PID file in place."; \
	exit 1

ingest: start_db
	cd src/ocr && \
	$(PYTHON) pipeline_cloudonly.py && \
	cd ../.. && \
	$(PYTHON) -m src.database.ingest_all
