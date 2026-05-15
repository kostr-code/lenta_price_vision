.PHONY: help setup dev demo ci-up obs down logs ps smoke clean rebuild backend-logs ml-logs db-shell

help:
	@echo "Available commands:"
	@echo "  make setup        - install local Python dependencies"
	@echo "  make dev          - start development environment"
	@echo "  make demo         - start demo environment"
	@echo "  make ci-up        - start CI-like environment"
	@echo "  make obs          - start app with observability"
	@echo "  make down         - stop services"
	@echo "  make logs         - show all logs"
	@echo "  make backend-logs - show backend logs"
	@echo "  make ml-logs      - show ML logs"
	@echo "  make ps           - show containers"
	@echo "  make smoke        - run smoke test"
	@echo "  make clean        - stop and remove volumes"
	@echo "  make rebuild      - rebuild all containers"
	@echo "  make db-shell     - open PostgreSQL shell"

setup:
	uv sync
	cd packages/ml && uv sync

dev:
	docker compose \
		-f docker-compose.yml \
		-f infra/compose/docker-compose.dev.yml \
		up --build

demo:
	docker compose \
		-f docker-compose.yml \
		-f infra/compose/docker-compose.demo.yml \
		up --build

ci-up:
	docker compose \
		-f docker-compose.yml \
		-f infra/compose/docker-compose.ci.yml \
		up -d --build

obs:
	docker compose \
		-f docker-compose.yml \
		-f infra/compose/docker-compose.observability.yml \
		up --build

down:
	docker compose down --remove-orphans

logs:
	docker compose logs -f

backend-logs:
	docker compose logs -f backend

ml-logs:
	docker compose logs -f ml

ps:
	docker compose ps

smoke:
	bash scripts/smoke_test.sh

clean:
	docker compose down -v --remove-orphans

rebuild:
	docker compose build --no-cache

db-shell:
	docker compose exec postgres psql -U price_audit -d price_audit
