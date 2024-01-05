"""script for downloading the necessary artifacts for the repository"""

import zipfile

from google.cloud import storage


def download_public_file(bucket_name, source_blob_name, destination_file_name) -> None:
    """Downloads a public blob from the bucket. taken from https://cloud.google.com/storage/docs/samples/storage-download-public-file"""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    print(
        f"Downloading public blob {source_blob_name} from bucket {bucket.name} to {destination_file_name}."
    )

    blob.download_to_filename(destination_file_name)


def main():
    for artifact in ["models", "data"]:
        # download artifact from the cloud
        download_public_file(
            "aro-agroptic-public", f"PETIT-GAN/demo/{artifact}.zip", f"./{artifact}.zip"
        )

        # unpacking
        print(f"unpacking {artifact} zipfile")
        with zipfile.ZipFile(artifact + ".zip", "r") as zip_ref:
            zip_ref.extractall(artifact)
    print("Download complete!")

if __name__ == "__main__":
    main()
