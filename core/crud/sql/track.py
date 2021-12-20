from core.models.track import Track
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy import func, union, distinct, desc
from core.mysql_database_connection.sqlalchemy_create_engine import SQLALCHEMY_DATABASE_URI

from core.crud.sqlalchemy import get_compiled_raw_mysql

engine = create_engine(SQLALCHEMY_DATABASE_URI)
db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))


def get_track_wiki(trackid: tuple):
    track_wiki = (db_session.query(Track.id,
                                   Track.title,
                                   Track.artist,
                                   func.json_extract(Track.info, "$.wiki_url").label("wiki_url"),
                                   func.json_extract(Track.info, "$.wiki.brief").label("wiki_content")
                                   )
                  .select_from(Track)
                  .filter(Track.id.in_(trackid))
                  )
    return track_wiki


def get_track_lyric(trackid: tuple):
    track_lyrics = (db_session.query(Track.id,
                                     Track.title,
                                     Track.artist,
                                     Track.lyrics
                                     )
                    .select_from(Track)
                    .filter(Track.id.in_(trackid))
                    )
    return track_lyrics


def get_all_by_track_ids(trackids: list):
    return db_session.query(Track).filter((Track.valid == 1),
                                          Track.id.in_(trackids)).order_by(
        Track.created_at.desc()).all()


def get_one_track_by_id(track_id: str):
    db_track =  db_session.query(Track).filter((Track.valid == 1),
                                          Track.id == track_id).all()
    if not db_track:
        return db_track
    else:
        return db_track[0]


if __name__ == "__main__":
    # k = get_compiled_raw_mysql(get_one_by_ids('1D379486D15C4D748056B3DE0D9D6249'))
    # print(k)

    trackid = 'joy'
    db_track = get_one_track_by_id(trackid)
    if not db_track:
        print("joy xinh")
    else:
        print(db_track.title)

        # print(db_track.title)

    # print(db_track.title)

    # for info in db_track:
    #     print(info)

#     print(db_tracks.id)
# for db_track in db_tracks:
#     print(db_track.id)
#     print(db_track.ext)
