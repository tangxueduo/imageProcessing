# import time
import zipfile
from io import BytesIO, StringIO

# import numpy as np
import pandas as pd
import pyzipper

# import pyminizip
from fastapi import FastAPI, Response

# from fastapi.response import FileResponse, Response
from starlette.responses import FileResponse, StreamingResponse

app = FastAPI()


class OctetStreamResponse(Response):
    media_type = "application/octet-stream"


@app.get("/")
def save_data():
    data = {"b": [1], "a": [0], "c": [2]}
    df = pd.DataFrame.from_dict(data)
    print(df)
    # df.to_csv("./test.csv")
    # # -------------------- Method 1 -------
    # compression_opts = dict(method="zip", archive_name="test.csv")
    # # # TODO: 确认存储 path， 增删改查操作
    # df.to_csv("test.zip", index=False, compression=compression_opts)
    # encrypt_status = encrypt_zip("test.zip")
    # print(encrypt_status)
    # ----------------------- Method 2 -------
    io_buffer = BytesIO()
    # 存的时候保证格式
    df.to_csv(io_buffer)
    io_buffer.seek(0)
    print(io_buffer.getvalue())
    # compression_level = 5
    # pyminizip.compress("test.csv", None, "test.zip", "tuixiang", compression_level)
    zip_io = BytesIO()
    secret_password = b"tuixiang"
    with pyzipper.AESZipFile(zip_io, "w", compression=pyzipper.ZIP_LZMA) as zf:
        zf.setpassword(secret_password)
        zf.setencryption(pyzipper.WZ_AES, nbits=128)
        zf.writestr("test.csv", io_buffer.getvalue())
    # with zipfile.ZipFile(zip_io, mode='w', compression=zipfile.ZIP_DEFLATED) as temp_zip:
    #     # for fpath in filenames:
    #     fpath = "test.csv"
    #     zip_path = "test.zip"
    #     # Add file, at correct path
    #     temp_zip.setpassword(b"tuixiang")
    #     temp_zip.writestr(fpath, io_buffer.getvalue())
    return StreamingResponse(
        iter([zip_io.getvalue()]),
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": f"attachment; filename=test.zip"},
    )


def encrypt_zip(zip_path: str, password: str = "tuixiang") -> str:
    """password-protected zip 文件
    # param: zip_path,zip 文件存储 path
    Returns:
            200, 500...
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as file:
            # 'setpassword' method is used to give a password to the 'Zip';
            # file.setpassword(pwd=bytes(password, encoding="utf-8"))
            file.extractall(pwd=b"{password}")
    except Exception as ex:
        raise ex
    return "success"


def decrypt_zip(zip_path: str, password: str = "tuixiang"):
    """根据密码解压 zip 文件"""
    pass
