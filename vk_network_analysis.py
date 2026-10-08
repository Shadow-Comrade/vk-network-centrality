import os
import re
import sys
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

            # Очистка от любых вариантов ссылок: https://vk.com/, http://vk.com/, vk.com/, m.vk.com/
            clean = re.sub(r"^(https?://)?(m\.)?vk\.com/", "", raw, flags=re.IGNORECASE)
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
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл со списком студентов не найден: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    ids = []

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
        # Берем первый столбец
        first_col = df.columns[0]
        ids = df[first_col].dropna().astype(str).str.strip().tolist()
    elif ext == ".csv":
        df = pd.read_csv(file_path)
        first_col = df.columns[0]
        ids = df[first_col].dropna().astype(str).str.strip().tolist()
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    ids.append(line)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {ext}. Используйте .txt, .csv или .xlsx")

    logger.info(f"Загружено записей из файла {file_path}: {len(ids)}")
    return ids


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

    return G, student_meta


def build_vk_network(
    client: VKClient,
    group_users: Dict[int, Dict[str, Any]],
    include_friends_of_friends: bool = True,
    max_fof_per_friend: int = 150
) -> nx.Graph:
    """
    Собирает граф:
    1. Друзья членов группы.
    2. Связи между членами группы и их друзьями.
    3. (Опционально) Друзья друзей с контролем объема выборки.
    """
    G = nx.Graph()

    # Добавляем членов группы
    for uid, meta in group_users.items():
        G.add_node(uid, is_group_member=True, label=meta["full_name"], domain=meta["domain"])

    group_ids = set(group_users.keys())
    student_friends: Dict[int, Set[int]] = {}

    logger.info("--- Шаг 1/2: Сбор друзей для всех членов группы ---")
    all_first_hop_friends: Set[int] = set()

    for uid in tqdm(group_ids, desc="Сбор друзей одногруппников"):
        friends = client.get_user_friends(uid)
        student_friends[uid] = friends
        for friend_id in friends:
            G.add_node(friend_id, is_group_member=(friend_id in group_ids))
            G.add_edge(uid, friend_id)
            all_first_hop_friends.add(friend_id)

    logger.info(f"Собрано уникальных друзей 1-го круга: {len(all_first_hop_friends)}")

    # Шаг 2: Друзья друзей (2-й уровень)
    if include_friends_of_friends:
        logger.info("--- Шаг 2/2: Сбор связей между друзьями и друзьями друзей ---")
        # Для баланса скорости VK API: берем друзей, которые дружат хотя бы с 2 студентами,
        # либо выборку наиболее активных узлов, чтобы не делать 10 000 долгих запросов
        mutual_or_priority_friends = [
            f for f in all_first_hop_friends
            if f not in group_ids and sum(1 for s in group_ids if f in student_friends[s]) >= 2
        ]

        # Если общих мало, берем срез первых 30-50 друзей
        if len(mutual_or_priority_friends) < 30:
            sample_candidates = list(all_first_hop_friends - group_ids)[:40]
            mutual_or_priority_friends = list(set(mutual_or_priority_friends + sample_candidates))

        logger.info(f"Сбор 2-го круга для {len(mutual_or_priority_friends)} ключевых друзей...")
        for friend_id in tqdm(mutual_or_priority_friends, desc="Друзья друзей"):
            fof = client.get_user_friends(friend_id)
            # Добавляем ребра только к уже известным узлам или ограничиваем количество новых
            for f_node in list(fof)[:max_fof_per_friend]:
                G.add_node(f_node, is_group_member=(f_node in group_ids))
                G.add_edge(friend_id, f_node)

    logger.info(f"Итоговый граф сформирован: Узлов = {G.number_of_nodes()}, Ребер = {G.number_of_edges()}")
    return G


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
                "Количество связей (Degree)": deg_val,
                "Посредничество (Betweenness)": round(b_val, 6),
                "Близость (Closeness)": round(c_val, 6),
                "Собственный вектор (Eigenvector)": round(e_val, 6)
            })
        else:
            results.append({
                "VK_ID": uid,
                "Имя Фамилия": meta.get("full_name", f"id{uid}"),
                "Профиль": f"https://vk.com/id{uid}",
                "Количество связей (Degree)": 0,
                "Посредничество (Betweenness)": 0.0,
                "Близость (Closeness)": 0.0,
                "Собственный вектор (Eigenvector)": 0.0
            })

    df = pd.DataFrame(results)
    # Сортируем по влиятельности (например, по посредничеству)
    df = df.sort_values(by="Посредничество (Betweenness)", ascending=False).reset_index(drop=True)
    return df


def export_results(df: pd.DataFrame, G: nx.Graph, output_prefix: str = "centrality_report"):
    """Экспорт результатов анализа в Excel, CSV и граф в формате GEXF."""
    csv_file = f"{output_prefix}.csv"
    xlsx_file = f"{output_prefix}.xlsx"
    gexf_file = f"{output_prefix}.gexf"

    df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    df.to_excel(xlsx_file, index=False)
    logger.info(f"Табличные отчеты сохранены в: {csv_file}, {xlsx_file}")

    # Сохраняем граф для программы Gephi (очень ценится преподавателями)
    try:
        # Приводим типы атрибутов для Gephi
        G_export = G.copy()
        for _, data in G_export.nodes(data=True):
            for k, v in data.items():
                data[k] = str(v)
        nx.write_gexf(G_export, gexf_file)
        logger.info(f"Файл графа для Gephi сохранен: {gexf_file}")
    except Exception as e:
        logger.warning(f"Не удалось экспортировать .gexf: {e}")


def visualize_network(G: nx.Graph, group_ids: Set[int], output_image: str = "vk_graph_plot.png"):
    """
    Отрисовывает граф, выделяя членов группы ярким цветом и увеличенным размером,
    аналогично иллюстрации на лекционном слайде.
    """
    logger.info(f"Построение визуализации графа ({G.number_of_nodes()} узлов)...")
    plt.figure(figsize=(14, 14), dpi=300)

    # Используем алгоритм пружинного размещения (Spring Layout)
    pos = nx.spring_layout(G, k=0.15, iterations=40, seed=42)

    # Разделяем узлы на группу и остальных
    group_nodes = [n for n in G.nodes() if n in group_ids]
    other_nodes = [n for n in G.nodes() if n not in group_ids]

    # 1. Фоновые узлы (друзья / друзья друзей)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=other_nodes,
        node_size=25,
        node_color="#64B5F6",
        alpha=0.45,
        label="Друзья и связи"
    )

    # 2. Узлы членов группы
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=group_nodes,
        node_size=280,
        node_color="#E53935",
        edgecolors="#FFFFFF",
        linewidths=1.5,
        label="Одногруппники"
    )

    # 3. Ребра сети
    nx.draw_networkx_edges(
        G, pos,
        alpha=0.12,
        edge_color="#78909C",
        width=0.6
    )

    # 4. Подписи только для членов группы
    labels = {}
    for n in group_nodes:
        name = G.nodes[n].get("label") or f"id{n}"
        labels[n] = name.split()[0] if " " in name else name

    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=8,
        font_family="sans-serif",
        font_weight="bold",
        font_color="#212121"
    )

    plt.title("Социальный граф группы VK: друзья и друзья друзей", fontsize=16, pad=20, fontweight="bold")
    plt.axis("off")
    plt.legend(scatterpoints=1, loc="upper right", frameon=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_image, bbox_inches="tight")
    plt.close()
    logger.info(f"Изображение графа успешно сохранено: {output_image}")


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
        "--no-fof",
        action="store_true",
        help="Отключить сбор друзей друзей (быстрый режим: только 1-й круг)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  АНАЛИЗ ЦЕНТРАЛЬНОСТЕЙ СОЦИАЛЬНОЙ СЕТИ ГРУППЫ (VK API + NetworkX)")
    print("=" * 70)

    # Демонстрационный режим или работа с реальным VK API
    if args.demo or not args.token:
        if not args.token and not args.demo:
            logger.warning("Токен VK не передан (--token или VK_TOKEN). Запуск в DEMO-режиме!")
        G, group_users = generate_demo_network()
    else:
        raw_ids = load_student_ids(args.input)
        client = VKClient(access_token=args.token)

        logger.info("Сопоставление идентификаторов с пользователями VK...")
        group_users = client.resolve_user_ids(raw_ids)

        if not group_users:
            logger.error("Не удалось найти ни одного пользователя. Проверьте входной файл.")
            sys.exit(1)

        G = build_vk_network(
            client=client,
            group_users=group_users,
            include_friends_of_friends=(not args.no_fof)
        )

    # Расчет требуемых центральностей
    group_ids_set = set(group_users.keys())
    df_metrics = calculate_group_centralities(G, group_users)

    # Вывод результатов в консоль
    print("\n" + "#" * 70)
    print("  РЕЗУЛЬТАТЫ ОЦЕНКИ ЦЕНТРАЛЬНОСТЕЙ ДЛЯ ЧЛЕНОВ ГРУППЫ:")
    print("#" * 70)
    print(df_metrics.to_string(index=False))
    print("#" * 70 + "\n")

    # Экспорт результатов
    export_results(df_metrics, G, output_prefix="centrality_report")
    visualize_network(G, group_ids_set, output_image="vk_graph_plot.png")

    print("[✔] Выполнение успешно завершено!")
    print("    - Таблица метрик: centrality_report.xlsx и centrality_report.csv")
    print("    - Графическое изображение: vk_graph_plot.png")
    print("    - Файл графа для Gephi: centrality_report.gexf")


if __name__ == "__main__":
    main()