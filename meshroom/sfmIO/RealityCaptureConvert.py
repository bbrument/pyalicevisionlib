"""
Meshroom nodes for converting between RealityCapture XMP and AliceVision SfMData formats.

Two nodes are provided:
- RCToSfMData: Converts RealityCapture XMP camera files to AliceVision SfMData JSON.
- SfMDataToRC: Converts AliceVision SfMData JSON to RealityCapture XMP camera files.
"""

from meshroom.core import desc

from pyalicevisionlib.scripts.rc_to_sfmdata import convert_rc_to_sfmdata
from pyalicevisionlib.scripts.sfmdata_to_rc import convert_sfmdata_to_rc


class RCToSfMData(desc.Node):
    """Convert RealityCapture XMP camera files to AliceVision SfMData.

    Reads a folder of RealityCapture XMP files (one per image) and a folder of
    the corresponding images, then produces an AliceVision SfMData JSON file
    that can be used directly in a Meshroom pipeline.

    Coordinate conventions are handled automatically:
    - RC rotation (world2cam, row-major) is converted to AV rotation (cam2world).
    - RC principal point (normalised by max dimension) is converted to AV pixel offset.
    - World coordinate axes are corrected to match AliceVision/Meshroom conventions.
    """

    category = "SfmIO"

    inputs = [
        desc.File(
            name="xmpFolder",
            label="XMP Folder",
            description="Folder containing RealityCapture XMP files (one per image).",
            value="",
        ),
        desc.File(
            name="imagesFolder",
            label="Images Folder",
            description="Folder containing the source images referenced by the XMP files.",
            value="",
        ),
        desc.File(
            name="referenceSfmData",
            label="Reference SfMData",
            description=(
                "Optional path to a reference SfMData file. "
                "When provided, view IDs are matched by image name instead of being generated."
            ),
            value="",
        ),
        desc.FloatParam(
            name="sensorWidth",
            label="Sensor Width (mm)",
            description="Physical sensor width in millimetres used to convert the 35 mm-equivalent focal length.",
            value=36.0,
            range=(1.0, 100.0, 0.1),
        ),
        desc.FloatParam(
            name="sensorHeight",
            label="Sensor Height (mm)",
            description="Physical sensor height in millimetres.",
            value=24.0,
            range=(1.0, 100.0, 0.1),
        ),
        desc.StringParam(
            name="cameraMake",
            label="Camera Make",
            description="Camera manufacturer name stored in the output SfMData metadata.",
            value="Unknown",
        ),
        desc.StringParam(
            name="cameraModel",
            label="Camera Model",
            description="Camera model name stored in the output SfMData metadata.",
            value="Unknown",
        ),
        desc.StringParam(
            name="serialNumber",
            label="Serial Number",
            description="Camera serial number used to group views into intrinsic groups.",
            value="0",
        ),
    ]

    outputs = [
        desc.File(
            name="output",
            label="SfMData",
            description="Path to the output AliceVision SfMData JSON file.",
            value="{nodeCacheFolder}/sfmData.json",
        ),
    ]

    def process(self, node):
        reference = node.referenceSfmData.value if node.referenceSfmData.value else None

        convert_rc_to_sfmdata(
            xmp_folder=node.xmpFolder.value,
            images_folder=node.imagesFolder.value,
            output_path=node.output.value,
            sensor_width=node.sensorWidth.value,
            sensor_height=node.sensorHeight.value,
            camera_make=node.cameraMake.value,
            camera_model=node.cameraModel.value,
            serial_number=node.serialNumber.value,
            reference_sfmdata=reference,
        )


class SfMDataToRC(desc.Node):
    """Convert AliceVision SfMData to RealityCapture XMP camera files.

    Reads an AliceVision SfMData file (.json / .sfm / .abc) and writes one XMP
    file per view into the specified output folder.  Images can optionally be
    copied alongside the XMP files so that the folder is immediately importable
    into RealityCapture.

    Coordinate conventions are handled automatically:
    - AV rotation (cam2world) is converted to RC rotation (world2cam, row-major).
    - AV principal point (pixel offset) is converted to RC normalised value.
    - World coordinate axes are corrected to match RealityCapture conventions.
    """

    category = "SfmIO"

    inputs = [
        desc.File(
            name="input",
            label="SfMData",
            description="Path to the input AliceVision SfMData file (.json, .sfm or .abc).",
            value="",
        ),
        desc.StringParam(
            name="imagesFolderName",
            label="Images Sub-Folder Name",
            description=(
                "Name of the sub-folder inside the output folder where images are copied. "
                "Only used when 'Copy Images' is enabled."
            ),
            value="images",
        ),
        desc.BoolParam(
            name="copyImages",
            label="Copy Images",
            description="Copy the source images into the output folder next to the XMP files.",
            value=True,
        ),
        desc.FloatParam(
            name="sensorWidth",
            label="Sensor Width (mm)",
            description=(
                "Default sensor width in millimetres used to convert focal length to the "
                "35 mm-equivalent format expected by RealityCapture. "
                "The value stored in the SfMData is used when available."
            ),
            value=36.0,
            range=(1.0, 100.0, 0.1),
        ),
    ]

    outputs = [
        desc.File(
            name="output",
            label="Output Folder",
            description="Folder containing the generated XMP files (and optionally the copied images).",
            value="{nodeCacheFolder}/rc_export",
        ),
    ]

    def process(self, node):
        convert_sfmdata_to_rc(
            sfmdata_path=node.input.value,
            output_folder=node.output.value,
            images_folder_name=node.imagesFolderName.value,
            copy_images=node.copyImages.value,
            sensor_width=node.sensorWidth.value,
        )
