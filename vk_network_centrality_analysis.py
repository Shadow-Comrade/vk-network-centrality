import os
import re
import sys
import json
import time
import argparse
import logging
from typing import List, Dict, Set, Tuple, Optional, Any

import requests
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("VKGraphAnalyzer")

VK_API_VERSION = "5.199"
VK_API_BASE_URL = "https://api.vk.com/method/"
# VK limits requests to 3 per second for user/service tokens
DEFAULT_REQUEST_DELAY = 0.35

# Файл-снимок собранной сети. Позволяет не обращаться к VK API повторно.
DEFAULT_SNAPSHOT = "vk_network_snapshot.json"
SNAPSHOT_FORMAT_VERSION = 1


def save_snapshot(
    path: str,
    G: nx.Graph,
    group_users: Dict[int, Dict[str, Any]],
    student_friends: Dict[int, Set[int]],
    fof_lists: Dict[int, Set[int]],
    source_ids: List[str]
) -> None:
    """
    Сохраняет всё собранное из VK в один JSON-файл.

    Что попадает в снимок:
      - graph           : все рёбра (и узлы с атрибутами);
      - group_users     : метаданные одногруппников (имя, домен, профиль);
      - student_friends : списки друзей 1-го круга по каждому студенту;
      - fof_lists       : списки друзей 2-го круга по каждому опрошенному узлу
                          (это «сырьё», из которого строятся кросс-связи);
      - source_ids      : исходный список идентификаторов из students.txt.

    Наличие student_friends и fof_lists важно: если завтра у кого-то появится
    новый друг, можно пересобрать связи, не опрашивая заново всех остальных.
    """
    payload = {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_ids": list(source_ids),
        "group_users": {str(k): v for k, v in group_users.items()},
        "nodes": [
            {"id": int(n), "attrs": {k: str(v) for k, v in d.items()}}
            for n, d in G.nodes(data=True)
        ],
        "edges": [[int(u), int(v)] for u, v in G.edges()],
        "student_friends": {str(k): sorted(int(x) for x in v) for k, v in student_friends.items()},
        "fof_lists": {str(k): sorted(int(x) for x in v) for k, v in fof_lists.items()},
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    logger.info(
        f"Снимок сети сохранён: {path} ({size_mb:.1f} МБ) — "
        f"{len(payload['nodes'])} узлов, {len(payload['edges'])} рёбер, "
        f"{len(payload['student_friends'])} списков 1-го круга, "
        f"{len(payload['fof_lists'])} списков 2-го круга."
    )


def load_snapshot(path: str) -> Tuple[nx.Graph, Dict[int, Dict[str, Any]], Dict[int, Set[int]], Dict[int, Set[int]], List[str]]:
    """Читает снимок сети из JSON и восстанавливает граф и списки друзей."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Снимок сети не найден: {path}\n"
            f"  Сначала соберите данные из VK: --update"
        )

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    version = payload.get("format_version")
    if version != SNAPSHOT_FORMAT_VERSION:
        raise ValueError(
            f"Несовместимый формат снимка: {version} (ожидается {SNAPSHOT_FORMAT_VERSION}). "
            f"Пересоберите данные командой --update."
        )

    G = nx.Graph()
    for node in payload["nodes"]:
        attrs = dict(node.get("attrs", {}))
        # is_group_member возвращаем к bool, остальное оставляем строками
        if "is_group_member" in attrs:
            attrs["is_group_member"] = str(attrs["is_group_member"]).lower() == "true"
        G.add_node(int(node["id"]), **attrs)
    for u, v in payload["edges"]:
        G.add_edge(int(u), int(v))

    group_users = {int(k): v for k, v in payload["group_users"].items()}
    student_friends = {int(k): set(v) for k, v in payload["student_friends"].items()}
    fof_lists = {int(k): set(v) for k, v in payload["fof_lists"].items()}
    source_ids = list(payload.get("source_ids", []))

    logger.info(
        f"Снимок загружен: {path} (сохранён {payload.get('saved_at', '?')})\n"
        f"  Узлов: {G.number_of_nodes()}, рёбер: {G.number_of_edges()}, "
        f"одногруппников: {len(group_users)}."
    )
    return G, group_users, student_friends, fof_lists, source_ids


class VKClient:
    """
    Класс для безопасного взаимодействия с VK API с контролем частоты запросов
    и обработкой типичных ошибок (приватные профили, удаленные страницы, капча).
    """
    def __init__(self, access_token: str, delay: float = DEFAULT_REQUEST_DELAY):
        self.access_token = access_token
        self.delay = delay
        self.session = requests.Session()

    def call_method(self, method_name: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Вызов произвольного метода VK API с задержкой и обработкой ошибок."""
        payload = {
            **params,
            "access_token": self.access_token,
            "v": VK_API_VERSION
        }
        url = f"{VK_API_BASE_URL}{method_name}"
        time.sleep(self.delay)

        try:
            response = self.session.post(url, data=payload, timeout=15)
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                err = data["error"]
                err_code = err.get("error_code")
                err_msg = err.get("error_msg")
                # Выводим понятное описание ошибки в консоль
                logger.warning(f"Ошибка VK API [{err_code}]: {err_msg}")
                return None

            return data.get("response")
        except requests.exceptions.RequestException as e:
            logger.error(f"Сетевая ошибка при обращении к {method_name}: {e}")
            return None

    def resolve_user_ids(self, identifiers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Преобразует список screen_name, domain, ссылок или id в числовые VK ID
        и возвращает метаданные (имя, фамилия).
        """
        clean_ids = []
        cyrillic_detected = []

        for raw in identifiers:
            raw = str(raw).strip().strip('"').strip("'")
            if not raw:
                continue

            # Проверка: если в строке русские буквы (ФИО вместо ссылки/ID)
            if re.search(r"[а-яА-ЯёЁ]", raw):
                cyrillic_detected.append(raw)
                continue

            # Очистка от любых вариантов ссылок: vk.com, vk.ru, m.vk.com, www.vk.com и т.д.
            clean = re.sub(r"^(https?://)?(m\.|www\.)?vk\.(com|ru)/", "", raw, flags=re.IGNORECASE)
            # Отсекаем параметры запросов (?all=1...) и якоря (#...)
            clean = clean.split("?")[0].split("#")[0]
            clean = clean.replace("@", "").strip("/").strip()

            if clean:
                clean_ids.append(clean)

        if cyrillic_detected:
            logger.error(
                f"В students.txt обнаружены русские имена/слова: {cyrillic_detected}\n"
                "  -> VK API не умеет искать по имени и фамилии текстом!\n"
                "  -> Укажите именно ссылки (например, vk.com/durov) или короткие id/логины."
            )

        resolved_users = {}
        if not clean_ids:
            return resolved_users

        logger.info(f"Распознанные идентификаторы для VK: {clean_ids}")

        # Запрашиваем пачками до 100 человек за раз
        chunk_size = 100
        for i in range(0, len(clean_ids), chunk_size):
            chunk = clean_ids[i:i + chunk_size]
            res = self.call_method("users.get", {
                "user_ids": ",".join(chunk),
                "fields": "domain,is_closed,deactivated"
            })
            if res:
                for user_info in res:
                    uid = user_info["id"]
                    first_name = user_info.get("first_name", "")
                    last_name = user_info.get("last_name", "")
                    domain = user_info.get("domain", f"id{uid}")
                    deactivated = user_info.get("deactivated")
                    is_closed = user_info.get("is_closed", False)

                    resolved_users[uid] = {
                        "id": uid,
                        "domain": domain,
                        "full_name": f"{first_name} {last_name}".strip(),
                        "is_valid": deactivated is None,
                        "is_closed": is_closed
                    }
            else:
                logger.warning(f"Не удалось получить данные от VK для: {chunk}")

        logger.info(f"Успешно сопоставлено пользователей: {len(resolved_users)} из {len(clean_ids)}")
        return resolved_users

    def get_user_friends(self, user_id: int) -> Set[int]:
        """Возвращает множество ID друзей пользователя."""
        res = self.call_method("friends.get", {
            "user_id": user_id,
            "count": 5000
        })
        if res and isinstance(res, dict) and "items" in res:
            return set(res["items"])
        return set()


def load_student_ids(file_path: str) -> List[str]:
    """
    Загружает идентификаторы из текстового файла (.txt)
    или таблицы Excel/CSV (.xlsx, .xls, .csv).

    Для Excel/CSV: столбец определяется автоматически — берётся первый столбец,
    содержащий пригодные для VK идентификаторы (ссылки, логины или числовые ID),
    что позволяет пропускать строки-заголовки вида «Ссылка на ВК:».
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл со списком студентов не найден: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    ids: List[str] = []

    def looks_like_vk_id(value: str) -> bool:
        """Ссылка vk.com/vk.ru, логин или числовой ID."""
        if not value:
            return False
        if re.search(r"(?:^|//)(?:m\.|www\.)?vk\.(?:com|ru)/", value, re.IGNORECASE):
            return True
        if re.fullmatch(r"\d+", value):
            return True
        # Логин: латиница/цифры/._ , без пробелов и кириллицы
        return bool(re.fullmatch(r"[A-Za-z0-9._]{2,}", value))

    if ext in [".xlsx", ".xls", ".xlsm"]:
        df = pd.read_excel(file_path, header=None)
        best_col, best_count = None, 0
        for col in df.columns:
            values = [
                str(v).strip() for v in df[col].dropna().tolist()
                if str(v).strip() and looks_like_vk_id(str(v).strip())
            ]
            if len(values) > best_count:
                best_col, best_count = col, len(values)
        if best_col is None:
            raise ValueError(
                f"В файле {file_path} не найдено ни одной колонки со ссылками/ID ВКонтакте."
            )
        ids = [
            str(v).strip() for v in df[best_col].dropna().tolist()
            if str(v).strip() and looks_like_vk_id(str(v).strip())
        ]
        logger.info(f"В таблице {file_path} выбрана колонка '{best_col}' ({len(ids)} записей).")
    elif ext == ".csv":
        df = pd.read_csv(file_path, header=None)
        first_col = df.columns[0]
        ids = [
            str(v).strip() for v in df[first_col].dropna().tolist()
            if str(v).strip()
        ]
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ids.append(line)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {ext}. Используйте .txt, .csv или .xlsx")

    # Убираем дубликаты, сохраняя исходный порядок
    seen: Set[str] = set()
    unique_ids: List[str] = []
    for raw in ids:
        key = raw.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique_ids.append(raw.strip())

    logger.info(f"Загружено записей из файла {file_path}: {len(unique_ids)} (без дублей)")
    return unique_ids


def generate_demo_network() -> Tuple[nx.Graph, Dict[int, Dict[str, Any]]]:
    """
    Генерирует реалистичный демонстрационный граф (если нет токена VK),
    чтобы код можно было протестировать и защитить сразу.
    """
    logger.info("Генерация тестовой сети одногруппников (Demo Mode)...")
    import random

    num_students = 15
    student_meta = {}
    for i in range(1, num_students + 1):
        uid = 1000 + i
        student_meta[uid] = {
            "id": uid,
            "domain": f"student_{i}",
            "full_name": f"Студент {i}",
            "is_valid": True,
            "is_closed": False
        }

    # Генерируем связный случайный граф группы
    G = nx.erdos_renyi_graph(n=num_students, p=0.35, seed=42)
    mapping = {node: 1000 + (node + 1) for node in G.nodes()}
    G = nx.relabel_nodes(G, mapping)

    # Добавляем общих внешних друзей (друзья и друзья друзей)
    external_friends = [2000 + j for j in range(1, 40)]
    for ext_f in external_friends:
        # Каждый внешний друг дружит с 1-3 студентами
        connected_students = random.sample(list(student_meta.keys()), k=random.randint(1, 3))
        for s in connected_students:
            G.add_edge(s, ext_f)

    # Добавляем связи между внешними друзьями (друзья друзей)
    for _ in range(30):
        u, v = random.sample(external_friends, 2)
        G.add_edge(u, v)

    for n in G.nodes():
        G.nodes[n]["is_group_member"] = (n in student_meta)
        if n in student_meta:
            G.nodes[n]["label"] = student_meta[n]["full_name"]
            G.nodes[n]["domain"] = student_meta[n]["domain"]

    logger.info(
        f"Демо-граф создан: Точек (V) = {G.number_of_nodes()}, Граней (E) = {G.number_of_edges()}"
    )
    return G, student_meta


def build_vk_network(
    client: VKClient,
    group_users: Dict[int, Dict[str, Any]],
    include_friends_of_friends: bool = True,
    max_fof_per_friend: int = 150,
    max_fof_clique: int = 40
) -> nx.Graph:
    """
    Собирает полный социологический граф:
    1. Одногруппники и их прямые друзья (1-й круг).
    2. Связи МЕЖДУ самими друзьями 1-го круга (кросс-рёбра).
    3. Друзья друзей (2-й круг) и взаимные связи МЕЖДУ ними.
    """
    G = nx.Graph()

    # Добавляем членов группы
    for uid, meta in group_users.items():
        G.add_node(uid, is_group_member=True, label=meta["full_name"], domain=meta["domain"])

    group_ids = set(group_users.keys())
    student_friends: Dict[int, Set[int]] = {}
    # Списки друзей 2-го круга по каждому опрошенному узлу (сырьё для кросс-связей)
    fof_lists: Dict[int, Set[int]] = {}

    logger.info("--- Шаг 1/3: Сбор друзей 1-го круга для членов группы ---")
    all_first_hop_friends: Set[int] = set()

    for uid in tqdm(group_ids, desc="Сбор друзей 1-го круга"):
        friends = client.get_user_friends(uid)
        student_friends[uid] = friends
        for friend_id in friends:
            G.add_node(friend_id, is_group_member=(friend_id in group_ids))
            G.add_edge(uid, friend_id)
            all_first_hop_friends.add(friend_id)

    logger.info(f"Собрано уникальных друзей 1-го круга: {len(all_first_hop_friends)}")

    # Шаг 2 и 3: Сбор друзей друзей и поиск всех связей МЕЖДУ ними
    if include_friends_of_friends:
        logger.info("--- Шаг 2/3: Сбор друзей 2-го круга и связей между ними ---")
        
        # Выбираем мосты (дружат с 2+ студентами) и наиболее активные контакты
        priority_friends = [
            f for f in all_first_hop_friends
            if f not in group_ids and sum(1 for s in group_ids if f in student_friends[s]) >= 2
        ]

        if len(priority_friends) < 35:
            extra = list(all_first_hop_friends - group_ids)[:45]
            priority_friends = list(set(priority_friends + extra))

        logger.info(f"Опрос окружения для {len(priority_friends)} ключевых узлов...")

        fof_cache: Dict[int, Set[int]] = {}

        for friend_id in tqdm(priority_friends, desc="Друзья друзей и связи"):
            fof = client.get_user_friends(friend_id)
            fof_cache[friend_id] = fof
            
            # Добавляем ребра к друзьям друзей
            for f_node in list(fof)[:max_fof_per_friend]:
                G.add_node(f_node, is_group_member=(f_node in group_ids))
                G.add_edge(friend_id, f_node)

        logger.info("--- Шаг 3/3: Учет связей МЕЖДУ друзьями и друзьями друзей ---")
        # Строим обратный индекс: кто из опрошенных узлов дружит с данным узлом.
        # Это позволяет за O(сумма списков) найти все рёбра, а не перебором пар.
        friend_to_samplers: Dict[int, Set[int]] = {}
        for sampler_id, sampler_friends in fof_cache.items():
            for f_node in sampler_friends:
                friend_to_samplers.setdefault(f_node, set()).add(sampler_id)

        # 1) Рёбра между опрошенными узлами (1-й круг <-> 1-й круг)
        cross_edges = 0
        for sampler_id, sampler_friends in fof_cache.items():
            for other in sampler_friends:
                if other in fof_cache:
                    G.add_edge(sampler_id, other)
                    cross_edges += 1

        # 2) Рёбра МЕЖДУ друзьями 2-го круга: если два узла 2-го круга
        #    присутствуют в списке друзей одного и того же опрошенного узла,
        #    значит они дружат между собой.
        #    Ограничиваем веер, чтобы клики популярных людей не давали взрывной рост.
        fof_cross_edges = 0
        for sampler_id, sampler_friends in fof_cache.items():
            known = [f for f in sampler_friends if f in G]
            if len(known) > max_fof_clique:
                known = known[:max_fof_clique]
            for i in range(len(known)):
                u = known[i]
                for j in range(i + 1, len(known)):
                    v = known[j]
                    if not G.has_edge(u, v):
                        G.add_edge(u, v)
                        fof_cross_edges += 1

        logger.info(
            f"Кросс-связи: {cross_edges} между узлами 1-го круга, "
            f"{fof_cross_edges} между друзьями 2-го круга."
        )
        fof_lists = fof_cache

    logger.info(f"Итоговый граф сформирован: Точек (V) = {G.number_of_nodes()}, Граней (E) = {G.number_of_edges()}")
    return G, student_friends, fof_lists


def rebuild_edges_from_lists(
    G: nx.Graph,
    group_users: Dict[int, Dict[str, Any]],
    student_friends: Dict[int, Set[int]],
    fof_lists: Dict[int, Set[int]],
    max_fof_per_friend: int = 150,
    max_fof_clique: int = 40
) -> nx.Graph:
    """
    Пересобирает все рёбра графа из сохранённых списков друзей, НЕ обращаясь к VK.

    Используется, когда нужно применить новые правила связывания (например,
    изменить max_fof_clique) к уже собранным данным.
    """
    G.clear()
    group_ids = set(group_users.keys())

    for uid, meta in group_users.items():
        G.add_node(uid, is_group_member=True, label=meta["full_name"], domain=meta["domain"])

    # 1-й круг
    for uid, friends in student_friends.items():
        for friend_id in friends:
            G.add_node(friend_id, is_group_member=(friend_id in group_ids))
            G.add_edge(uid, friend_id)

    # 2-й круг
    for friend_id, fof in fof_lists.items():
        for f_node in list(fof)[:max_fof_per_friend]:
            G.add_node(f_node, is_group_member=(f_node in group_ids))
            G.add_edge(friend_id, f_node)

    # Кросс-связи 1-го круга между собой
    for sampler_id, sampler_friends in fof_lists.items():
        for other in sampler_friends:
            if other in fof_lists:
                G.add_edge(sampler_id, other)

    # Кросс-связи между друзьями 2-го круга
    for sampler_id, sampler_friends in fof_lists.items():
        known = [f for f in sampler_friends if f in G]
        if len(known) > max_fof_clique:
            known = known[:max_fof_clique]
        for i in range(len(known)):
            u = known[i]
            for j in range(i + 1, len(known)):
                v = known[j]
                if not G.has_edge(u, v):
                    G.add_edge(u, v)

    logger.info(
        f"Граф пересобран из снимка: Точек (V) = {G.number_of_nodes()}, "
        f"Граней (E) = {G.number_of_edges()}"
    )
    return G

def calculate_graph_statistics(G: nx.Graph) -> Dict[str, Any]:
    """
    Вычисляет строгие топологические параметры графа (точки, грани, плотность).
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    
    # Средняя степень вершины (среднее число граней на одну точку)
    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / num_nodes if num_nodes > 0 else 0.0

    return {
        "Количество точек (вершин |V|)": num_nodes,
        "Количество граней (рёбер |E|)": num_edges,
        "Плотность графа (Density)": round(density, 6),
        "Среднее число связей (Avg Degree)": round(avg_degree, 2)
    }

def calculate_group_centralities(
    G: nx.Graph,
    group_users: Dict[int, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Вычисляет три метрики центральности:
    1. По посредничеству (Betweenness Centrality)
    2. По близости (Closeness Centrality)
    3. Собственного вектора (Eigenvector Centrality)
    Фильтрует результат строго для членов группы.
    """
    logger.info("Вычисление метрик центральности на построенном графе...")

    # 1. Посредничество (Betweenness)
    logger.info("-> Расчет Betweenness Centrality...")
    betweenness = nx.betweenness_centrality(G, normalized=True)

    # 2. Близость (Closeness)
    logger.info("-> Расчет Closeness Centrality...")
    closeness = nx.closeness_centrality(G)

    # 3. Собственный вектор (Eigenvector)
    logger.info("-> Расчет Eigenvector Centrality...")
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=2000, tol=1e-06)
    except nx.PowerIterationFailedConvergence:
        logger.warning("Eigenvector centrality не сошлась стандартным методом, используем numpy-аппроксимацию.")
        try:
            eigenvector = nx.eigenvector_centrality_numpy(G)
        except Exception:
            eigenvector = {n: 0.0 for n in G.nodes()}

    # Степень узла (Degree) как полезная базовая метрика
    degree_dict = dict(G.degree())

    results = []
    for uid, meta in group_users.items():
        if uid in G:
            b_val = betweenness.get(uid, 0.0)
            c_val = closeness.get(uid, 0.0)
            e_val = eigenvector.get(uid, 0.0)
            deg_val = degree_dict.get(uid, 0)

            results.append({
                "VK_ID": uid,
                "Имя Фамилия": meta.get("full_name", f"id{uid}"),
                "Профиль": f"https://vk.com/{meta.get('domain', 'id' + str(uid))}",
                "Количество граней / связей (Degree)": deg_val,
                "Посредничество (Betweenness)": round(b_val, 6),
                "Близость (Closeness)": round(c_val, 6),
                "Собственный вектор (Eigenvector)": round(e_val, 6)
            })
        else:
            results.append({
                "VK_ID": uid,
                "Имя Фамилия": meta.get("full_name", f"id{uid}"),
                "Профиль": f"https://vk.com/id{uid}",
                "Количество граней / связей (Degree)": 0,
                "Посредничество (Betweenness)": 0.0,
                "Близость (Closeness)": 0.0,
                "Собственный вектор (Eigenvector)": 0.0
            })

    df = pd.DataFrame(results)
    df = df.sort_values(by="Посредничество (Betweenness)", ascending=False).reset_index(drop=True)
    return df

def export_results(
    df_metrics: pd.DataFrame,
    stats: Dict[str, Any],
    G: nx.Graph,
    output_prefix: str = "centrality_report"
):
    """Экспорт метрик и отдельного листа со статистикой точек и граней в Excel/CSV."""
    csv_file = f"{output_prefix}.csv"
    xlsx_file = f"{output_prefix}.xlsx"
    gexf_file = f"{output_prefix}.gexf"

    # Экспорт в CSV
    df_metrics.to_csv(csv_file, index=False, encoding="utf-8-sig")

    # Экспорт в Excel с двумя вкладками: Центральности и Топология (Точки/Грани)
    df_stats = pd.DataFrame([stats])
    with pd.ExcelWriter(xlsx_file, engine="openpyxl") as writer:
        df_metrics.to_excel(writer, sheet_name="Центральности_группы", index=False)
        df_stats.to_excel(writer, sheet_name="Топология_сети (V и E)", index=False)

    logger.info(f"Табличные отчеты (метрики + точки и грани) сохранены в: {csv_file}, {xlsx_file}")

    # Сохраняем граф для Gephi
    try:
        G_export = G.copy()
        for _, data in G_export.nodes(data=True):
            for k, v in data.items():
                data[k] = str(v)
        nx.write_gexf(G_export, gexf_file)
        logger.info(f"Файл графа для Gephi сохранен: {gexf_file}")
    except Exception as e:
        logger.warning(f"Не удалось экспортировать .gexf: {e}")

def visualize_network(
    G: nx.Graph,
    group_ids: Set[int],
    output_image: str = "vk_graph_plot.png",
    render_all: bool = True,
    max_friends_per_student: int = 35,
    max_fof_per_friend: int = 5,
    figsize_inches: int = 40,
    dpi: int = 300,
    spacing: float = 2.6,
    layout_iterations: int = 60
):
    """
    Создает визуализацию сети с поддержкой 100% охвата всех узлов.

    figsize_inches / dpi — физический размер и разрешение итогового изображения.
    spacing — множитель «пружинной» расталкивающей силы: чем больше значение,
            тем дальше узлы разнесены друг от друга (k в spring_layout).
    layout_iterations — число итераций укладки.

    Использует 4-уровневую палитру:
      🔴 Красный: Одногруппники
      🟢 Зеленый: Общие друзья (мосты между 2+ студентами)
      🟡 Желтый: Личные друзья (1-й круг)
      🔵 Голубой: Друзья друзей (2-й круг)
    """
    import math
    import matplotlib.patheffects as path_effects
    from matplotlib.lines import Line2D

    total_graph_nodes = G.number_of_nodes()
    logger.info(f"Подготовка визуализации графа (всего в сети: {total_graph_nodes} узлов)...")

    group_nodes = [n for n in G.nodes() if n in group_ids]

    # Сбор прямых друзей для каждого студента
    student_friends_map: Dict[int, Set[int]] = {}
    direct_friends_all: Set[int] = set()
    for s in group_nodes:
        s_friends = set(G.neighbors(s)) - group_ids
        student_friends_map[s] = s_friends
        direct_friends_all.update(s_friends)

    # 1. Общие друзья (связывают 2+ студентов)
    shared_friends_all = {
        f for f in direct_friends_all
        if sum(1 for s in group_nodes if f in student_friends_map[s]) >= 2
    }

    # 2. Личные друзья студентов (1-й круг)
    solo_friends_all = direct_friends_all - shared_friends_all

    # 3. Друзья друзей (2-й круг) — абсолютно все оставшиеся узлы сети
    fof_all = set(G.nodes()) - set(group_nodes) - direct_friends_all

    if render_all:
        # Режим полного отображения: берем абсолютно ВСЕ узлы сети без отсева
        selected_solo = solo_friends_all
        selected_fof = fof_all
        H = G.copy()
        logger.info(
            f"Полный режим: отображаем ВСЕ {H.number_of_nodes()} узлов "
            f"({len(group_nodes)} членов группы, {len(shared_friends_all)} общих, "
            f"{len(selected_solo)} прямых друзей, {len(selected_fof)} друзей друзей)."
        )
    else:
        # Сжатый режим выборки (для компактного превью)
        selected_solo = set()
        for s in group_nodes:
            s_solo = [n for n in G.neighbors(s) if n in solo_friends_all]
            s_solo_sorted = sorted(s_solo, key=lambda x: G.degree(x), reverse=True)
            selected_solo.update(s_solo_sorted[:max_friends_per_student])

        selected_fof = set()
        for f in (shared_friends_all | selected_solo):
            fof_cands = [n for n in G.neighbors(f) if n in fof_all]
            fof_sorted = sorted(fof_cands, key=lambda x: G.degree(x), reverse=True)
            selected_fof.update(fof_sorted[:max_fof_per_friend])

        keep_nodes = set(group_nodes) | shared_friends_all | selected_solo | selected_fof
        H = G.subgraph(keep_nodes).copy()

    logger.info(
        f"Расчет укладки графа пружинным алгоритмом "
        f"({H.number_of_nodes()} узлов, {layout_iterations} итераций)..."
    )
    # spacing — множитель расталкивания: больше значение => больше дистанция между узлами
    k_optimal = spacing / (H.number_of_nodes() ** 0.46)
    pos = nx.spring_layout(H, k=k_optimal, iterations=layout_iterations, seed=42)

    fig, ax = plt.subplots(figsize=(figsize_inches, figsize_inches), dpi=dpi)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FAFAFA")

    fof_edges = []        # рёбра 2-го круга (друзья друзей)
    fof_cross_edges = []  # кросс-связи МЕЖДУ друзьями 2-го круга
    direct_edges = []     # прямые связи 1-го круга
    core_edges = []       # связи студентов с мостами и между собой

    group_ids_set = set(group_ids)
    fof_set = set(selected_fof)
    for u, v in H.edges():
        u_is_fof = u in fof_set
        v_is_fof = v in fof_set
        if u_is_fof and v_is_fof:
            fof_cross_edges.append((u, v))
        elif u_is_fof or v_is_fof:
            fof_edges.append((u, v))
        elif (u in group_ids_set and v in shared_friends_all) or \
             (v in group_ids_set and u in shared_friends_all) or \
             (u in group_ids_set and v in group_ids_set):
            core_edges.append((u, v))
        else:
            direct_edges.append((u, v))

    logger.info(
        f"Ребра: 2-й круг {len(fof_edges)}, кросс-связи 2-го круга {len(fof_cross_edges)}, "
        f"1-й круг {len(direct_edges)}, ядро {len(core_edges)}."
    )

    # Слой 1a: Ребра между друзьями друзей (яркая бирюзовая паутина)
    if fof_cross_edges:
        nx.draw_networkx_edges(
            H, pos,
            edgelist=fof_cross_edges,
            ax=ax,
            alpha=0.55 if render_all else 0.75,
            edge_color="#06B6D4",
            width=0.55 if render_all else 1.1
        )

    # Слой 1b: Ребра друзей друзей к их «родителям» 1-го круга
    if fof_edges:
        nx.draw_networkx_edges(
            H, pos,
            edgelist=fof_edges,
            ax=ax,
            alpha=0.30 if render_all else 0.55,
            edge_color="#38BDF8",
            width=0.35 if render_all else 0.8
        )

    # Слой 2: Ребра прямых друзей (яркий оранжево-янтарный)
    if direct_edges:
        nx.draw_networkx_edges(
            H, pos,
            edgelist=direct_edges,
            ax=ax,
            alpha=0.55 if render_all else 0.75,
            edge_color="#F59E0B",
            width=0.8 if render_all else 1.3
        )

    # Слой 3: Акцентные ребра между студентами и их мостами (насыщенный малиновый)
    if core_edges:
        nx.draw_networkx_edges(
            H, pos,
            edgelist=core_edges,
            ax=ax,
            alpha=0.95,
            edge_color="#DB2777",
            width=2.4
        )

    # Уровень 4: Голубые (Друзья друзей — узлы 2-го круга)
    if selected_fof:
        nx.draw_networkx_nodes(
            H, pos,
            nodelist=list(selected_fof),
            node_size=14 if render_all else 55,
            node_color="#38BDF8",
            alpha=0.55 if render_all else 0.80,
            ax=ax
        )

    # Уровень 3: Желтые (Личные друзья студентов)
    if selected_solo:
        nx.draw_networkx_nodes(
            H, pos,
            nodelist=list(selected_solo),
            node_size=55 if render_all else 130,
            node_color="#FACC15",
            edgecolors="#FFFFFF",
            linewidths=0.9 if render_all else 1.4,
            alpha=0.92,
            ax=ax
        )

    # Уровень 2: Зеленые (Общие друзья / мосты)
    if shared_friends_all:
        nx.draw_networkx_nodes(
            H, pos,
            nodelist=list(shared_friends_all),
            node_size=340,
            node_color="#22C55E",
            edgecolors="#FFFFFF",
            linewidths=2.4,
            alpha=0.98,
            ax=ax
        )

    # Уровень 1: Красные (Члены группы — максимальный масштаб)
    degrees = dict(G.degree())
    student_sizes = [max(3000, min(degrees.get(uid, 50) * 18, 7000)) for uid in group_nodes]

    nx.draw_networkx_nodes(
        H, pos,
        nodelist=group_nodes,
        node_size=student_sizes,
        node_color="#EF4444",
        edgecolors="#FFFFFF",
        linewidths=5.0,
        alpha=1.0,
        ax=ax
    )

    labels = {}
    for uid in group_nodes:
        full_label = G.nodes[uid].get("label") or f"id{uid}"
        parts = full_label.split()
        labels[uid] = f"{parts[0]}\n{parts[1][:1]}." if len(parts) >= 2 else full_label

    text_items = nx.draw_networkx_labels(
        H, pos,
        labels=labels,
        font_size=20,
        font_family="sans-serif",
        font_weight="bold",
        font_color="#0F172A",
        ax=ax
    )

    for text_obj in text_items.values():
        text_obj.set_path_effects([
            path_effects.Stroke(linewidth=7.0, foreground="white"),
            path_effects.Normal()
        ])

    plt.title(
        f"Социальная сеть группы VK: Точек (вершин |V|) = {H.number_of_nodes()} | Граней (рёбер |E|) = {H.number_of_edges()}",
        fontsize=30,
        fontweight="bold",
        pad=34,
        color="#0F172A"
    )

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'Одногруппники ({len(group_nodes)})',
               markerfacecolor='#EF4444', markersize=22, markeredgecolor='#FFFFFF', markeredgewidth=2.0),
        Line2D([0], [0], marker='o', color='w', label=f'Общие друзья / мосты ({len(shared_friends_all)})',
               markerfacecolor='#22C55E', markersize=16, markeredgecolor='#FFFFFF', markeredgewidth=1.5),
        Line2D([0], [0], marker='o', color='w', label=f'Личные друзья 1-го круга ({len(selected_solo)})',
               markerfacecolor='#FACC15', markersize=13, markeredgecolor='#FFFFFF', markeredgewidth=1.0),
        Line2D([0], [0], marker='o', color='w', label=f'Друзья друзей 2-го круга ({len(selected_fof)})',
               markerfacecolor='#38BDF8', markersize=10, alpha=0.9, markeredgecolor='none'),
        Line2D([0], [0], color='#DB2777', lw=3.5,
               label=f'Связи студентов и мостов ({len(core_edges)})'),
        Line2D([0], [0], color='#F59E0B', lw=2.6,
               label=f'Прямые друзья 1-го круга ({len(direct_edges)})'),
        Line2D([0], [0], color='#06B6D4', lw=2.2,
               label=f'Связи между друзьями 2-го круга ({len(fof_cross_edges)})'),
        Line2D([0], [0], color='#38BDF8', lw=1.6,
               label=f'Друзья друзей ↔ 1-й круг ({len(fof_edges)})'),
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=19,
        borderpad=1.1,
        labelspacing=0.9
    )

    ax.text(
        0.02, 0.02,
        f"• ТОПОЛОГИЯ СЕТИ: Точек (вершин |V|) = {H.number_of_nodes()} | Граней (рёбер |E|) = {H.number_of_edges()}\n"
        "• Красные: члены группы (размер узла пропорционален степени влияния)\n"
        "• Зеленые: общие друзья (коммуникационные мосты между студентами)\n"
        "• Желтые: прямые друзья членов группы (1-й круг)\n"
        "• Голубые: друзья друзей (2-й круг)\n"
        f"• Малиновые линии: связи студентов и мостов ({len(core_edges)})\n"
        f"• Оранжевые линии: прямые связи 1-го круга ({len(direct_edges)})\n"
        f"• Бирюзовые линии: связи МЕЖДУ друзьями 2-го круга ({len(fof_cross_edges)})",
        transform=ax.transAxes,
        fontsize=18,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#CBD5E1", alpha=0.94)
    )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_image, bbox_inches="tight", dpi=dpi)
    plt.close()
    logger.info(
        f"Итоговая визуализация с {H.number_of_nodes()} узлами сохранена в: {output_image} "
        f"({figsize_inches}x{figsize_inches} дюймов @ {dpi} DPI)"
    )

def main():
    parser = argparse.ArgumentParser(
        description="Анализ центральностей социальной сети членов группы во ВКонтакте."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="students.txt",
        help="Путь к файлу со списком ID/ссылок одногруппников (.txt, .xlsx, .csv)"
    )
    parser.add_argument(
        "--token", "-t",
        type=str,
        default=os.getenv("VK_TOKEN", ""),
        help="Сервисный ключ доступа или Access Token VK API"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Запустить в демонстрационном режиме с синтетическими данными (без токена)"
    )
    parser.add_argument(
        "--snapshot",
        type=str,
        default=DEFAULT_SNAPSHOT,
        help=f"Путь к файлу-снимку собранной сети. По умолчанию {DEFAULT_SNAPSHOT}."
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Обратиться к VK API и обновить снимок. Без этого флага VK не опрашивается."
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Пересобрать граф из снимка (применить новые правила связывания без похода в VK)."
    )
    parser.add_argument(
        "--no-fof",
        action="store_true",
        help="Отключить сбор друзей друзей (быстрый режим: только 1-й круг)"
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Отрисовать только сжатую выборку узлов вместо всех 4800+"
    )
    parser.add_argument(
        "--figsize",
        type=int,
        default=40,
        help="Размер изображения в дюймах (сторона квадрата). По умолчанию 40."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Разрешение итогового изображения. По умолчанию 300."
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=2.6,
        help="Дистанция между узлами: множитель расталкивания. Больше => просторнее. По умолчанию 2.6."
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  АНАЛИЗ ЦЕНТРАЛЬНОСТЕЙ СОЦИАЛЬНОЙ СЕТИ ГРУППЫ (VK API + NetworkX)")
    print("=" * 70)

    # --- Получение данных: снимок на диске либо VK API ---
    group_users: Dict[int, Dict[str, Any]] = {}
    student_friends: Dict[int, Set[int]] = {}
    fof_lists: Dict[int, Set[int]] = {}
    source_ids: List[str] = []

    if args.demo:
        G, group_users = generate_demo_network()

    elif args.update:
        # Явный запрос к VK: собираем всё заново и сохраняем снимок
        if not args.token:
            logger.error("Для обновления из VK нужен токен (--token или VK_TOKEN).")
            sys.exit(1)

        source_ids = load_student_ids(args.input)
        client = VKClient(access_token=args.token)

        logger.info("Сопоставление идентификаторов с пользователями VK...")
        group_users = client.resolve_user_ids(source_ids)

        if not group_users:
            logger.error("Не удалось найти ни одного пользователя. Проверьте входной файл.")
            sys.exit(1)

        G, student_friends, fof_lists = build_vk_network(
            client=client,
            group_users=group_users,
            include_friends_of_friends=(not args.no_fof)
        )

        save_snapshot(
            path=args.snapshot,
            G=G,
            group_users=group_users,
            student_friends=student_friends,
            fof_lists=fof_lists,
            source_ids=source_ids
        )

    else:
        # Обычный режим: работаем из снимка, VK не опрашиваем
        try:
            G, group_users, student_friends, fof_lists, source_ids = load_snapshot(args.snapshot)
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

        if args.force_rebuild and fof_lists:
            logger.info("Пересборка графа из сохранённых списков друзей (без обращения к VK)...")
            G = rebuild_edges_from_lists(
                G=G,
                group_users=group_users,
                student_friends=student_friends,
                fof_lists=fof_lists
            )
            save_snapshot(
                path=args.snapshot,
                G=G,
                group_users=group_users,
                student_friends=student_friends,
                fof_lists=fof_lists,
                source_ids=source_ids
            )

    # 1. Расчет строгих топологических параметров (точки и грани)
    stats = calculate_graph_statistics(G)

    print("\n" + "=" * 70)
    print("  ТОПОЛОГИЧЕСКИЙ ПАСПОРТ ГРАФА (ТОЧКИ И ГРАНИ):")
    print("=" * 70)
    print(f"  • Точек (вершин, |V|):            {stats['Количество точек (вершин |V|)']}")
    print(f"  • Граней (рёбер, |E|):            {stats['Количество граней (рёбер |E|)']}")
    print(f"  • Плотность графа (Density):      {stats['Плотность графа (Density)']}")
    print(f"  • Среднее число связей на точку:  {stats['Среднее число связей (Avg Degree)']}")
    print("=" * 70 + "\n")

    # 2. Расчет требуемых центральностей
    group_ids_set = set(group_users.keys())
    df_metrics = calculate_group_centralities(G, group_users)

    # Вывод результатов в консоль
    print("#" * 70)
    print("  РЕЗУЛЬТАТЫ ОЦЕНКИ ЦЕНТРАЛЬНОСТЕЙ ДЛЯ ЧЛЕНОВ ГРУППЫ:")
    print("#" * 70)
    print(df_metrics.to_string(index=False))
    print("#" * 70 + "\n")

    # Экспорт результатов с отдельной вкладкой под точки и грани
    export_results(df_metrics, stats, G, output_prefix="centrality_report")
    visualize_network(
        G,
        group_ids_set,
        output_image="vk_graph_plot.png",
        render_all=(not args.sample_only),
        figsize_inches=args.figsize,
        dpi=args.dpi,
        spacing=args.spacing
    )

    print("[✔] Выполнение успешно завершено!")
    print("    - Таблица метрик и топологии: centrality_report.xlsx")
    print("    - Графическое изображение: vk_graph_plot.png")
    print("    - Файл графа для Gephi: centrality_report.gexf")
    if not args.demo:
        print(f"    - Снимок сети: {args.snapshot}")
        if not args.update:
            print("      (VK не опрашивался — данные взяты из снимка)")
            print(f"      Для обновления из VK: --update")


if __name__ == "__main__":
    main()