# Имя стека
STACK_NAME=mikoshi_stack
# Композ-файл
COMPOSE_FILE=docker-compose.yml

# Проверка, инициализирован ли Docker Swarm
check-swarm:
    @if [ -z "$$(docker info --format '{{.Swarm.LocalNodeState}}')" = "active" ]; then \
        echo "Инициализирую Docker Swarm..."; \
        docker swarm init; \
    else \
        echo "Docker Swarm уже инициализирован."; \
    fi

# Разворачиваем стек с всеми сервисами
deploy: check-swarm
    @echo "Разворачиваю стек $(STACK_NAME)..."
    docker stack deploy -c $(COMPOSE_FILE) $(STACK_NAME)

# Останавливаем все сервисы (удаление стека)
teardown:
    @echo "Останавливаю стек $(STACK_NAME)..."
    docker stack rm $(STACK_NAME)

# Обновляем текущее состояние сервиса (например, после изменения)
update:
    @echo "Обновляю стек $(STACK_NAME)..."
    docker stack deploy -c $(COMPOSE_FILE) $(STACK_NAME)

# Очистить docker volumes и networks
clean:
    @docker volume prune -f
    @docker network prune -f

# Просмотр статуса сервисов
status:
    @docker service ls

# Просмотр логов всех сервисов
logs-all:
    @docker service logs $(STACK_NAME)
