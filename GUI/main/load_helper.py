from .models import SessionDatabase, UploadedSessionDatabase
import os
from csv import reader as csv_reader
from django.conf import settings


def load_original_data(reset_upload_database):
    """
    Wipes the SessionDatabase and optionally wipes the UploadedSessionDatabase before loading
     the original dataset into the Django builtin database & detects their associated files
    :return: redirects to homepage with messages to notify the user for success or any errors
    """
    # Clear each database here
    SessionDatabase.objects.all().delete()
    if reset_upload_database == 'on':
        UploadedSessionDatabase.objects.all().delete()

    if not os.path.isfile(SessionDatabase.tsv_path):
        return {"success": False, "message": "Either this is not a file or the location is wrong!"}

    # Check if the tsv file exists
    expected_ordering = ['participant_id', 'session_id', 'gender', 'birth_age', 'birth_weight', 'singleton',
                         'scan_age', 'scan_number', 'radiology_score', 'sedation']

    found_mri_file_names = [f for f in os.listdir(SessionDatabase.default_mri_path) if f.endswith("nii")]
    found_vtps_files_names = [f for f in os.listdir(SessionDatabase.default_vtps_path)
                              if f.endswith("vtp") & f.find("inflated") != -1 & f.find("hemi-L") != -1]

    with open(SessionDatabase.tsv_path) as foo:
        reader = csv_reader(foo, delimiter='\t')
        for i, row in enumerate(reader):
            if i == 0:
                if row != expected_ordering:
                    return {"success": False,
                            "message": "FAILED! The table column names aren't what was expected or in the wrong order."}
                continue

            (participant_id, session_id, gender, birth_age, birth_weight, singleton, scan_age,
             scan_number, radiology_score, sedation) = row

            mri_file_path = next((f"{os.path.join(SessionDatabase.default_mri_path, x)}"
                                  for x in found_mri_file_names if (participant_id and session_id) in x), "")

            surface_file_path = next((f"{os.path.join(SessionDatabase.default_vtps_path, x)}"
                                      for x in found_vtps_files_names if (participant_id and session_id) in x), "")

            # Check for session ID uniqueness
            if SessionDatabase.objects.all().filter(session_id=session_id).count() > 0:
                print(f'tsv contains non-uniques session id: {session_id}')
                continue

            SessionDatabase.objects.create(participant_id=participant_id,
                                           session_id=int(session_id),
                                           gender=gender,
                                           birth_age=float(birth_age),
                                           birth_weight=float(birth_weight),
                                           singleton=singleton,
                                           scan_age=float(scan_age),
                                           scan_number=int(scan_number),
                                           radiology_score=radiology_score,
                                           sedation=sedation,
                                           mri_file=mri_file_path,
                                           surface_file=surface_file_path)

            # If an .nii file was found, correct the record.mri_file.name
            if mri_file_path != "":
                record = SessionDatabase.objects.get(session_id=session_id)
                record.mri_file.name = record.mri_file.name.split(settings.MEDIA_URL)[-1]
                record.save()

            # If an .nii file was found, correct the record.surface_file.name
            if surface_file_path != "":
                record = SessionDatabase.objects.get(session_id=session_id)
                record.surface_file.name = record.surface_file.name.split(settings.MEDIA_URL)[-1]
                record.save()

    return {"success": True, "message": "SUCCESS"}
