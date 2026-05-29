"""Profile management endpoints."""

import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.core.exceptions import CloudinaryUploadError
from app.core.rate_limit import limiter
from app.dependencies import CurrentUser, DBSession
from app.schemas.auth import UserResponse
from app.schemas.profile import DeleteProfileResponse, UpdateProfileRequest
from app.schemas.response import ErrorResponse, SuccessResponse
from app.services.profile import delete_profile, update_profile, update_profile_photo

logger = logging.getLogger(__name__)

router = APIRouter()

# Max file size: 10 MB
MAX_FILE_SIZE = 10 * 1024 * 1024

# Chunk size for reading files (8 KB)
CHUNK_SIZE = 8 * 1024

# Magic bytes (file signatures) for supported image formats
# These are the first few bytes that identify the file type
IMAGE_MAGIC_BYTES = {
    b"\xff\xd8\xff": "image/jpeg",  # JPEG
    b"\x89\x50\x4e\x47": "image/png",  # PNG
    b"\x47\x49\x46\x38": "image/gif",  # GIF (GIF87a and GIF89a)
}


def _validate_image_content(content: bytes) -> str:
    """Validate image content by checking magic bytes (file signature).

    Args:
        content: The file content bytes to validate.

    Returns:
        The MIME type of the validated image.

    Raises:
        HTTPException: If the content doesn't match supported image formats.
    """
    if len(content) < 4:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is too small to be a valid image",
        )

    # Check magic bytes to determine actual file type
    for magic_bytes, mime_type in IMAGE_MAGIC_BYTES.items():
        if content.startswith(magic_bytes):
            return mime_type

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Invalid file format. Only JPEG, PNG, and GIF are supported.",
    )


async def _read_file_with_size_check(file: UploadFile, max_size: int) -> bytes:
    """Read file in chunks while enforcing size limit.

    Reads the file in chunks and validates the total size does not exceed
    max_size. Fails early if size limit is exceeded to prevent memory exhaustion.

    Args:
        file: The uploaded file to read.
        max_size: Maximum allowed file size in bytes.

    Returns:
        The complete file content.

    Raises:
        HTTPException: If the file size exceeds the limit.
    """
    chunks: list[bytes] = []
    total_size = 0

    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break

        total_size += len(chunk)

        # Check size limit early to prevent memory exhaustion
        if total_size > max_size:
            max_mb = int(max_size / 1024 / 1024)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File size exceeds maximum allowed size of {max_mb} MB",
            )

        chunks.append(chunk)

    return b"".join(chunks)


@router.get(
    "",
    response_model=SuccessResponse[UserResponse],
    status_code=status.HTTP_200_OK,
    summary="Get current user profile",
    description="Returns the authenticated user's profile information.",
    responses={
        200: {"description": "Profile retrieved successfully"},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("10/minute")
async def get_profile(
    request: Request,
    current_user: CurrentUser,
) -> SuccessResponse[UserResponse]:
    """Retrieve the authenticated user's profile."""
    return SuccessResponse(
        message="Profile retrieved successfully",
        data=UserResponse.model_validate(current_user),
    )


@router.put(
    "",
    response_model=SuccessResponse[UserResponse],
    status_code=status.HTTP_200_OK,
    summary="Update user profile",
    description=(
        "Update profile information (first name, last name, email, and/or role). "
        "At least one field must be provided. "
        "Other fields remain unchanged."
    ),
    responses={
        200: {"description": "Profile updated successfully"},
        400: {"model": ErrorResponse, "description": "No fields to update"},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("5/minute")
async def update_user_profile(
    request: Request,
    session: DBSession,
    current_user: CurrentUser,
    payload: UpdateProfileRequest,
) -> SuccessResponse[UserResponse]:
    """Update the authenticated user's profile.

    Allows updating first_name, last_name, email, and/or role.
    At least one field must be provided to make a valid update request.
    """
    if (
        payload.first_name is None
        and payload.last_name is None
        and payload.email is None
        and payload.role is None
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one field must be provided",
        )

    updated_user = await update_profile(
        session=session,
        user=current_user,
        first_name=payload.first_name,
        last_name=payload.last_name,
        email=payload.email,
        role=payload.role,
    )

    logger.info("Updated profile for user %s", current_user.id)

    return SuccessResponse(
        message="Profile updated successfully",
        data=UserResponse.model_validate(updated_user),
    )


@router.delete(
    "",
    response_model=SuccessResponse[DeleteProfileResponse],
    status_code=status.HTTP_200_OK,
    summary="Delete user profile",
    description=(
        "Permanently delete the authenticated user's profile and all associated data. "
        "This action cannot be undone. The user's profile photo will also be removed "
        "from Cloudinary if it exists."
    ),
    responses={
        200: {"description": "Profile deleted successfully"},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("5/minute")
async def delete_user_profile(
    request: Request,
    session: DBSession,
    current_user: CurrentUser,
) -> SuccessResponse[DeleteProfileResponse]:
    """Delete the authenticated user's profile.

    Permanently removes the user account and all associated data,
    including the profile photo from Cloudinary if one exists.
    This action is irreversible.
    """
    user_id = current_user.id

    await delete_profile(
        session=session,
        user_id=user_id,
    )

    logger.info("Deleted profile for user %s", user_id)

    return SuccessResponse(
        message="Your profile has been permanently deleted.",
        data=DeleteProfileResponse(
            id=user_id,
        ),
    )


@router.put(
    "/photo",
    response_model=SuccessResponse[UserResponse],
    status_code=status.HTTP_200_OK,
    summary="Upload or update user profile photo",
    description=(
        "Upload a new profile photo. The image is uploaded to Cloudinary. "
        "If the user already has a profile photo, the old one is "
        "automatically deleted. "
        "Supports JPEG, PNG, and GIF formats. Max file size: 10 MB."
    ),
    responses={
        200: {"description": "Profile photo updated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid file or file too large"},
        401: {"model": ErrorResponse, "description": "Not authenticated"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
@limiter.limit("10/minute")
async def upload_profile_photo_endpoint(
    request: Request,
    session: DBSession,
    current_user: CurrentUser,
    file: UploadFile = File(  # noqa: B008
        ..., description="Image file (JPEG, PNG, GIF)"
    ),
) -> SuccessResponse[UserResponse]:
    """Upload or update the authenticated user's profile photo.

    Accepts image files and uploads them to Cloudinary.
    The previous profile photo is automatically deleted if it exists.

    Security measures:
    - File size is validated during streaming to prevent memory exhaustion
    - File content is validated by magic bytes to prevent spoofed formats
    """
    # Read file in chunks with size limit enforcement
    content = await _read_file_with_size_check(file, MAX_FILE_SIZE)

    # Validate file type by inspecting actual content (magic bytes)
    # This prevents clients from spoofing the content_type header
    actual_mime_type = _validate_image_content(content)

    logger.info(
        f"Processing profile photo upload for user {current_user.id}: "
        f"claimed={file.content_type}, actual={actual_mime_type}"
    )

    try:
        updated_user = await update_profile_photo(
            session=session,
            user=current_user,
            photo_data=content,
        )
    except CloudinaryUploadError as exc:
        logger.exception(
            "Cloudinary upload failed for user %s: %s", current_user.id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to upload image to Cloudinary",
        ) from exc

    logger.info("Updated profile photo for user %s", current_user.id)

    return SuccessResponse(
        message="Profile photo updated successfully",
        data=UserResponse.model_validate(updated_user),
    )
