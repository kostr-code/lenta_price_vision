.PHONY: dev demo ci-up obs down logs ps smoke clean

dev:
	docker compose -f docker-compose.yml -f infra/compose/docker-compose.dev.yml up --build

demo:
	docker compose -f docker-compose.yml -f infra/compose/docker-compose.demo.yml up --build

ci-up:
	docker compose -f docker-compose.yml -f infra/compose/docker-compose.ci.yml up -d --build

obs:
	docker compose -f docker-compose.yml -f infra/compose/docker-compose.observability.yml up --build

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

smoke:
	bash scripts/smoke_test.sh

clean:
	docker compose down -v --remove-orphans