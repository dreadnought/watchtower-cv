import time
import datetime

from b2sdk.account_info.in_memory import InMemoryAccountInfo
from b2sdk.api import B2Api


class BackblazeB2:
    def __init__(self, application_key_id, application_key, bucket_name, logger):
        self.logger = logger
        info = InMemoryAccountInfo()
        self.b2_api = B2Api(info)
        self.b2_api.authorize_account("production", application_key_id, application_key)

        self.bucket = self.b2_api.get_bucket_by_name(bucket_name)
    def upload_bytes(self, data_bytes, remote_file_name):
        self.logger.info("uploading %s" % remote_file_name)
        start_time = time.time()
        self.bucket.upload_bytes(
            data_bytes=data_bytes,
            file_name=remote_file_name,
        )
        self.logger.info("took %0.2f sec" % (time.time() - start_time))
        return self._get_url_with_auth(remote_file_name)

    def upload_file(self, local_file_name, remote_file_name):
        self.logger.info("uploading %s" % local_file_name)
        start_time = time.time()
        response = self.bucket.upload_local_file(
            local_file=local_file_name,
            file_name=remote_file_name,
        )
        self.logger.info("took %0.2f sec" % (time.time() - start_time))
        return self._get_url_with_auth(remote_file_name)

    def _get_url_with_auth(self, file_name):
        auth_token = self.bucket.get_download_authorization(file_name_prefix=file_name, valid_duration_in_seconds=60*60*24)
        base_url = self.bucket.get_download_url(file_name)
        url = f"{base_url}?Authorization={auth_token}"
        self.logger.debug(url)
        return url

    def cleanup_bucket(self, max_days):
        self.logger.info("cleanup starting")
        top_level = self.bucket.ls()
        today = datetime.datetime.now()
        for dir, date_str in top_level:
            dt = datetime.datetime.strptime(date_str, "%Y%m%d/")
            dt_diff = today - dt
            days_old = dt_diff.days
            if days_old < max_days:
                continue

            self.logger.info(f"{date_str} is {days_old} days old, deleting...")
            files = self.bucket.ls(folder_to_list=date_str)
            for f, name in files:
                file_meta = f.as_dict()
                self.logger.info(f"deleting {file_meta['fileName']}")
                self.bucket.delete_file_version(file_id=file_meta["fileId"], file_name=file_meta["fileName"])
        self.logger.info("cleanup done")
