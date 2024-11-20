"""
title: Knowledge Graph Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for creating and utilizing a knowledge graph from data.
requirements: pandas, networkx, plotly
"""
from typing import List, Union, Generator, Iterator
import os
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class Pipeline:
    def __init__(self):
        self.graph = None  # Будет хранить ваш граф знаний
        self.pos = None    # Позиции узлов для визуализации

    async def on_startup(self):
        """
        Выполняется при запуске сервера.
        Здесь мы загружаем данные, очищаем их и создаем граф знаний.
        """
        directory = '/app/backend/data'
        df = self.read_parquet_files(directory)
        if df.empty:
            print("В указанной директории нет данных.")
            return

        df = self.clean_dataframe(df)
        if df.empty:
            print("После очистки данных не осталось.")
            return

        self.graph = self.create_knowledge_graph(df)
        if self.graph.number_of_nodes() > 0:
            print(f"Граф успешно создан с {self.graph.number_of_nodes()} узлами и {self.graph.number_of_edges()} ребрами.")
            # Создаем 3D расположение узлов один раз для дальнейшей визуализации
            self.pos = nx.spring_layout(self.graph, dim=3)
        else:
            print("Граф пуст. Невозможно визуализировать.")
        pass

    async def on_shutdown(self):
        # Эта функция вызывается при остановке сервера.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Обрабатывает входящее сообщение пользователя и возвращает ответ.
        В этом примере мы будем возвращать информацию об узле, запрошенном пользователем.
        """
        # Предполагаем, что пользователь запрашивает информацию об определенном узле.
        node = user_message.strip()
        if self.graph is None or self.graph.number_of_nodes() == 0:
            return "Граф не инициализирован. Пожалуйста, убедитесь, что данные загружены правильно."

        if node in self.graph.nodes:
            # Получаем информацию об узле
            neighbors = list(self.graph.neighbors(node))
            response = f"Узел '{node}' найден. Связанные узлы: {', '.join(neighbors)}."
        else:
            response = f"Узел '{node}' не найден в графе."

        return response

    # Ниже приведены ваши функции, адаптированные как методы класса
    def read_parquet_files(self, directory):
        """
        Читает все Parquet-файлы в указанной директории и объединяет их в один DataFrame.
        """
        dataframes = []
        for filename in os.listdir(directory):
            if filename.endswith('.parquet'):
                file_path = os.path.join(directory, filename)
                df = pd.read_parquet(file_path)
                dataframes.append(df)
        return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

    def clean_dataframe(self, df):
        """
        Очищает DataFrame, удаляя строки с отсутствующими значениями в столбцах 'source' и 'target'.
        Преобразует столбцы 'source' и 'target' в строковый тип.
        """
        df = df.dropna(subset=['source', 'target'])
        df['source'] = df['source'].astype(str)
        df['target'] = df['target'].astype(str)
        return df

    def create_knowledge_graph(self, df):
        """
        Создает граф знаний из DataFrame.
        """
        G = nx.DiGraph()
        for _, row in df.iterrows():
            source = row['source']
            target = row['target']
            attributes = {k: v for k, v in row.items() if k not in ['source', 'target']}
            G.add_edge(source, target, **attributes)
        return G

    def visualize_graph_plotly(self):
        """
        Создает интерактивную 3D визуализацию графа знаний с дополнительными графиками.
        """
        if self.graph is None or self.graph.number_of_nodes() == 0:
            print("Граф пуст. Нечего визуализировать.")
            return

        G = self.graph
        pos = self.pos

        # Получение трассировок
        edge_trace, node_trace = self.create_node_link_trace(G, pos)
        edge_labels = nx.get_edge_attributes(G, 'relation')
        edge_label_trace = self.create_edge_label_trace(G, pos, edge_labels)

        # Создание дополнительных графиков
        degree_dist_fig = self.create_degree_distribution(G)
        centrality_fig = self.create_centrality_plot(G)

        # Создание подграфиков
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.7, 0.3],
            row_heights=[0.7, 0.3],
            specs=[
                [{"type": "scene", "rowspan": 2}, {"type": "xy"}],
                [None, {"type": "xy"}]
            ],
            subplot_titles=("3D Граф знаний", "Распределение степеней узлов", "Распределение степени центральности")
        )

        # Добавление трассировок в фигуру
        fig.add_trace(edge_trace, row=1, col=1)
        fig.add_trace(node_trace, row=1, col=1)
        fig.add_trace(edge_label_trace, row=1, col=1)
        fig.add_trace(degree_dist_fig.data[0], row=1, col=2)
        fig.add_trace(centrality_fig.data[0], row=2, col=2)

        # Настройка 3D сцены
        fig.update_layout(
            scene=dict(
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                zaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                aspectmode='cube'
            ),
            scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        )

        # Добавление интерактивных элементов (кнопок и слайдеров)
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(args=[{"visible": [True, True, True, True, True]}], label="Показать все", method="update"),
                        dict(args=[{"visible": [True, True, False, True, True]}], label="Скрыть метки ребер",
                             method="update"),
                        dict(args=[{"visible": [False, True, False, True, True]}], label="Только узлы", method="update")
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.05,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                ),
            ],
            sliders=[dict(
                active=0,
                currentvalue={"prefix": "Размер узла: "},
                pad={"t": 50},
                steps=[dict(method='update',
                            args=[{'marker.size': [i] * len(G.nodes)}],
                            label=str(i)) for i in range(5, 21, 5)]
            )]
        )

        fig.show()

    # Дополнительные методы для визуализации
    def create_node_link_trace(self, G, pos):
        """
        Создает 3D-трассировки для узлов и ребер графа.
        """
        # Создание координат для ребер
        edge_x = []
        edge_y = []
        edge_z = []
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_z.extend([z0, z1, None])
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        # Создание координат для узлов
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_z = [pos[node][2] for node in G.nodes()]
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                )
            )
        )

        # Настройка цвета узлов в зависимости от количества связей
        node_adjacencies = []
        node_text = []
        for node, adjacencies in G.adjacency():
            node_adjacencies.append(len(adjacencies))
            node_text.append(f'Узел: {node}<br>Количество связей: {len(adjacencies)}')
        node_trace.marker.color = node_adjacencies
        node_trace.text = node_text

        return edge_trace, node_trace

    def create_edge_label_trace(self, G, pos, edge_labels):
        """
        Создает 3D-трассировку для отображения меток ребер.
        """
        return go.Scatter3d(
            x=[(pos[edge[0]][0] + pos[edge[1]][0]) / 2 for edge in edge_labels],
            y=[(pos[edge[0]][1] + pos[edge[1]][1]) / 2 for edge in edge_labels],
            z=[(pos[edge[0]][2] + pos[edge[1]][2]) / 2 for edge in edge_labels],
            mode='text',
            text=list(edge_labels.values()),
            textposition='middle center',
            hoverinfo='none'
        )

    def create_degree_distribution(self, G):
        """
        Создает гистограмму распределения степеней узлов.
        """
        degrees = [d for n, d in G.degree()]
        fig = px.histogram(x=degrees, nbins=20, labels={'x': 'Степень', 'y': 'Количество'})
        fig.update_layout(
            title_text='Распределение степеней узлов',
            margin=dict(l=0, r=0, t=30, b=0),
            height=300
        )
        return fig

    def create_centrality_plot(self, G):
        """
        Создает коробчатую диаграмму распределения центральности узлов.
        """
        centrality = nx.degree_centrality(G)
        centrality_values = list(centrality.values())
        fig = px.box(y=centrality_values, labels={'y': 'Центральность'})
        fig.update_layout(
            title_text='Распределение степени центральности',
            margin=dict(l=0, r=0, t=30, b=0),
            height=300
        )
        return fig
