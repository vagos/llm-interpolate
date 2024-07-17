import click
import llm
import numpy as np
import sqlite_utils
from scipy.spatial.distance import cdist

def linear(start_point, end_point, embeddings, n):
    def lerp(embedding_1, embedding_2, n):
        return [(1 - alpha) * embedding_1 + alpha * embedding_2 for alpha in np.linspace(0, 1, n)]

    _, start_embedding = start_point
    end_id, end_embedding = end_point

    all_embeddings = np.array(list(embeddings.values()))
    all_ids = list(embeddings.keys())

    inbetween_points = lerp(start_embedding, end_embedding, n)
    selected_points = []

    for point in inbetween_points:
        distances = cdist([point], all_embeddings, 'cosine')[0]
        for idx in np.argsort(distances):
            nearest_point = all_ids[idx]
            if nearest_point in selected_points:
                continue
            selected_points.append(nearest_point)
            break

    # end_id must always be last
    if selected_points[-1] != end_id:
        selected_points.remove(end_id)
        selected_points.append(end_id)

    return selected_points


@llm.hookimpl
def register_commands(cli):
    @cli.command()
    @click.argument("collection")
    @click.argument("start_id")
    @click.argument("end_id")
    @click.option("-n",
                  default=10,
                  type=int,
                  help="Number of points between the start and end embedding")
    @click.option(
        "--method",
        default="linear",
        help="Method of interpolation between start and end embeddings",
    )
    @click.option(
        "-d",
        "--database",
        type=click.Path(
            file_okay=True, allow_dash=False, dir_okay=False, writable=True
        ),
        envvar="LLM_EMBEDDINGS_DB",
        help="SQLite database file containing embeddings",
    )
    def interpolate(collection, start_id, end_id, n, method, database):
        """
        Interpolate between embeddings in a collection

        Example usage, to interpolate between the embeddings of two songs with
        10 points.

        \b
            llm interpolate my_song_collection "RapSong.wav" "CountrySong.wav" -n 10

        Outputs a JSON array of collection ids
        """
        interpolation_methods = {
            "linear": linear
        }

        interpolation_method = interpolation_methods[method]
        if database:
            db = sqlite_utils.Database(database)
        else:
            db = sqlite_utils.Database(llm.user_dir() / "embeddings.db")
        rows = [
            (row[0], llm.decode(row[1]))
            for row in db.execute(
                """
            select id, embedding from embeddings
            where collection_id = (
                select id from collections where name = ?
            )
        """,
                [collection],
            ).fetchall()
        ]

        embeddings = {id: np.array(embedding) for id, embedding in rows}

        start_point = (start_id, embeddings[start_id])
        end_point = (end_id, embeddings[end_id])

        interpolated_points = interpolation_method(start_point, end_point, embeddings, n)
        last_idx = len(interpolated_points) - 1
        click.echo("[")
        for idx, point_id in enumerate(interpolated_points):
            if idx == last_idx:
                click.echo(f'    "{point_id}"')
            else:
                click.echo(f'    "{point_id}",')
        click.echo("]")
