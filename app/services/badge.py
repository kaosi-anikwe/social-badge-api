import logging
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

from sqlalchemy import case, func, select
from sqlalchemy import update as sa_update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import (
    BadgeAlreadyPublishedError,
    BadgeNotFoundError,
    CloudinaryUploadError,
    NotBadgeOwnerError,
    PlatformTemplateNotActiveError,
    PlatformTemplateNotFoundError,
    PublicBadgeNotFoundError,
)
from app.core.slug import generate_share_slug
from app.models import Badge, BadgeHashtag, PlatformTemplate
from app.services.cloudinary import (
    delete_logo,
    upload_logo,
)

logger = logging.getLogger(__name__)

# Valid gallery categories — used for validation so we can return a clean 400
# rather than an empty list when the client sends a typo.
VALID_CATEGORIES = frozenset(
    {
        "festivals",
        "hackathons",
        "conferences",
        "community",
        "bootcamp",
        "meetups",
        "speakers",
        "summit",
        "trending",
    }
)


async def _increment_template_badge_count(
    session: AsyncSession, platform_template_id: UUID
) -> None:
    await session.execute(
        sa_update(PlatformTemplate)
        .where(PlatformTemplate.id == platform_template_id)
        .values(total_badges_made=PlatformTemplate.total_badges_made + 1)
    )


async def create_badge(
    session: AsyncSession,
    organiser_id: UUID,
    platform_template_id: UUID,
) -> Badge:
    result = await session.execute(
        select(PlatformTemplate).where(
            PlatformTemplate.id == platform_template_id,
        )
    )
    platform_template = result.scalars().first()
    if platform_template is None:
        raise PlatformTemplateNotFoundError

    if not platform_template.is_active:
        raise PlatformTemplateNotActiveError

    instance = Badge(
        organiser_id=organiser_id,
        platform_template_id=platform_template_id,
        title=platform_template.title,
        canvas_data=platform_template.canvas_data or {},
    )
    session.add(instance)
    await session.flush()
    await _increment_template_badge_count(session, platform_template_id)

    await session.commit()
    await session.refresh(instance)

    logger.info(
        "Created template instance %s for organiser %s",
        instance.id,
        organiser_id,
    )
    return instance


async def publish_badge(
    session: AsyncSession,
    organiser_id: UUID,
    id: UUID,
) -> Badge:
    result = await session.execute(
        select(Badge).where(
            Badge.id == id,
            Badge.deleted_at.is_(None),
        )
    )
    badge = result.scalars().first()

    if not badge:
        raise BadgeNotFoundError
    if badge.organiser_id != organiser_id:
        raise NotBadgeOwnerError
    if badge.is_published:
        raise BadgeAlreadyPublishedError

    badge.is_published = True
    badge.published_at = datetime.now(UTC)

    if not badge.share_slug:
        badge.share_slug = generate_share_slug()

    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise

    await session.refresh(badge)

    logger.info(
        "Published badge %s by organiser %s",
        badge.id,
        organiser_id,
    )

    return badge


async def unpublish_badge(
    session: AsyncSession,
    organiser_id: UUID,
    id: UUID,
) -> Badge:
    result = await session.execute(select(Badge).where(Badge.id == id))
    template = result.scalars().first()
    if template is None or template.deleted_at is not None:
        raise BadgeNotFoundError
    if template.organiser_id != organiser_id:
        raise NotBadgeOwnerError

    template.is_published = False
    template.published_at = None
    await session.commit()
    await session.refresh(template)

    logger.info("Unpublished template %s by organiser %s", template.id, organiser_id)
    return template


async def upload_badge_logo(
    session: AsyncSession,
    id: UUID,
    organiser_id: UUID,
    image_data: bytes,
) -> str:
    """Upload a logo for a template instance and return the Cloudinary URL.

    Raises:
        BadgeNotFoundError: if the instance does not exist.
        NotBadgeOwnerError: if the instance belongs to another organiser.
        CloudinaryUploadError: if the Cloudinary upload fails.
    """
    result = await session.execute(
        select(Badge).where(
            Badge.id == id,
            Badge.deleted_at.is_(None),
        )
    )
    instance = result.scalars().first()

    if instance is None:
        raise BadgeNotFoundError

    if instance.organiser_id != organiser_id:
        raise NotBadgeOwnerError

    old_public_id = instance.logo_public_id

    # Upload first so the DB always points at a live asset.
    logo_url, public_id = await upload_logo(image_data)

    instance.logo_url = logo_url
    instance.logo_public_id = public_id
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        # DB commit failed — best-effort cleanup of the just-uploaded asset.
        try:
            await delete_logo(public_id)
        except Exception:
            logger.warning(
                "Failed to clean up Cloudinary asset %s after DB commit failure",
                public_id,
            )
        raise
    await session.refresh(instance)

    # Only delete the old asset after the DB is consistent.
    # A failure here is non-fatal — the new logo is already persisted.
    if old_public_id:
        try:
            await delete_logo(old_public_id)
        except Exception:
            logger.warning(
                "Failed to delete old Cloudinary asset %s — manual cleanup may be required",  # noqa: E501
                old_public_id,
            )

    logger.info(
        "Uploaded logo for template instance %s (public_id=%s)",
        id,
        public_id,
    )
    return logo_url


async def get_public_badge_by_slug(
    session: AsyncSession,
    slug: str,
) -> Badge:
    result = await session.execute(
        select(Badge)
        .options(selectinload(Badge.hashtags))
        .where(
            Badge.share_slug == slug,
            Badge.is_published.is_(True),
            Badge.deleted_at.is_(None),
        )
    )
    template = result.scalars().first()
    if template is None:
        raise PublicBadgeNotFoundError

    logger.info("Public lookup for slug %s resolved to template %s", slug, template.id)
    return template


_PUBLIC_WHERE = (
    Badge.is_published.is_(True),
    Badge.deleted_at.is_(None),
)


async def increment_badge_share_count(session: AsyncSession, slug: str) -> None:
    """Atomically increment share_count for a published badge.

    Called after a successful public page fetch; silently no-ops if the row
    has since disappeared (best-effort counter — accuracy is not critical).
    """
    await session.execute(
        sa_update(Badge)
        .where(Badge.share_slug == slug, *_PUBLIC_WHERE)
        .values(share_count=Badge.share_count + 1)
    )
    await session.commit()


async def increment_badge_creation_count(session: AsyncSession, slug: str) -> None:
    """Atomically increment creation_count for a published badge.

    Raises PublicBadgeNotFoundError when the slug does not resolve to a
    published, non-deleted badge so the router can return 404.
    """
    result = cast(
        CursorResult[Any],
        await session.execute(
            sa_update(Badge)
            .where(Badge.share_slug == slug, *_PUBLIC_WHERE)
            .values(creation_count=Badge.creation_count + 1)
        ),
    )
    await session.commit()
    if result.rowcount == 0:
        raise PublicBadgeNotFoundError


async def list_platform_templates(
    session: AsyncSession,
    category: str | None = None,
    page: int = 1,
    limit: int = 10,
) -> tuple[list[PlatformTemplate], int]:
    if category is not None:
        normalised = category.strip().lower()
        if normalised not in VALID_CATEGORIES:
            raise ValueError(
                f"Unknown category '{category}'. "
                f"Valid options: {', '.join(sorted(VALID_CATEGORIES))}"
            )
        count_stmt = select(func.count(PlatformTemplate.id)).where(
            PlatformTemplate.is_active.is_(True),
            PlatformTemplate.category == normalised,
        )
        stmt = (
            select(PlatformTemplate)
            .where(
                PlatformTemplate.is_active.is_(True),
                PlatformTemplate.category == normalised,
            )
            .order_by(PlatformTemplate.title)
        )
    else:
        count_stmt = select(func.count(PlatformTemplate.id)).where(
            PlatformTemplate.is_active.is_(True)
        )
        stmt = (
            select(PlatformTemplate)
            .where(PlatformTemplate.is_active.is_(True))
            .order_by(PlatformTemplate.category.nulls_last(), PlatformTemplate.title)
        )

    count_result = await session.execute(count_stmt)
    total = count_result.scalar_one()

    offset = (page - 1) * limit
    stmt = stmt.offset(offset).limit(limit)

    result = await session.execute(stmt)
    templates = list(result.scalars().all())

    logger.debug(
        "list_platform_templates: category=%s page=%d limit=%d "
        "returned %d of %d total results",
        category,
        page,
        limit,
        len(templates),
        total,
    )
    return templates, total


async def get_platform_template(
    session: AsyncSession,
    id: UUID,
) -> PlatformTemplate:
    result = await session.execute(
        select(PlatformTemplate).where(
            PlatformTemplate.id == id,
            PlatformTemplate.is_active.is_(True),
        )
    )
    template = result.scalars().first()
    if template is None:
        raise PlatformTemplateNotFoundError

    return template


async def duplicate_badge(
    session: AsyncSession,
    organiser_id: UUID,
    id: UUID,
) -> Badge:
    result = await session.execute(
        select(Badge)
        .options(selectinload(Badge.hashtags))
        .where(
            Badge.id == id,
            Badge.deleted_at.is_(None),
        )
    )
    original = result.scalars().first()
    if original is None:
        raise BadgeNotFoundError

    if original.organiser_id != organiser_id:
        raise NotBadgeOwnerError

    copy = Badge(
        organiser_id=organiser_id,
        platform_template_id=original.platform_template_id,
        title=f"{original.title} (Copy)",
        canvas_data=original.canvas_data,
        default_caption=original.default_caption,
        destination_link=original.destination_link,
        thumbnail_url=original.thumbnail_url,
        logo_url=None,
        logo_public_id=None,
        access_type=original.access_type,
        is_published=False,
        share_slug=None,
        published_at=None,
    )
    session.add(copy)
    await session.flush()

    for tag in original.hashtags:
        session.add(BadgeHashtag(badge_id=copy.id, hashtag=tag.hashtag))

    await _increment_template_badge_count(session, original.platform_template_id)

    await session.commit()
    await session.refresh(copy)

    logger.info(
        "Duplicated template %s as %s for organiser %s",
        id,
        copy.id,
        organiser_id,
    )
    return copy


async def list_badges(
    session: AsyncSession,
    organiser_id: UUID,
    page: int = 1,
    limit: int = 20,
) -> tuple[list[Badge], int]:
    base_conditions = (
        Badge.organiser_id == organiser_id,
        Badge.deleted_at.is_(None),
    )

    count_result = await session.execute(
        select(func.count(Badge.id)).where(*base_conditions)
    )
    total = count_result.scalar_one()

    stmt = (
        select(Badge)
        .where(*base_conditions)
        .order_by(
            Badge.updated_at.desc().nulls_last(),
            Badge.created_at.desc().nulls_last(),
        )
        .offset((page - 1) * limit)
        .limit(limit)
    )
    result = await session.execute(stmt)
    templates = list(result.scalars().all())

    logger.debug(
        "list_badges: organiser=%s page=%d limit=%d returned %d of %d total",
        organiser_id,
        page,
        limit,
        len(templates),
        total,
    )
    return templates, total


async def delete_badge(
    session: AsyncSession,
    organiser_id: UUID,
    id: UUID,
) -> None:
    result = await session.execute(
        select(Badge).where(
            Badge.id == id,
            Badge.deleted_at.is_(None),
        )
    )
    template = result.scalars().first()
    if template is None:
        raise BadgeNotFoundError
    if template.organiser_id != organiser_id:
        raise NotBadgeOwnerError

    logo_public_id = template.logo_public_id

    template.deleted_at = datetime.now(UTC)
    template.is_published = False
    template.published_at = None
    await session.commit()

    logger.info(
        "Deleted organiser template %s (organiser=%s)",
        id,
        organiser_id,
    )

    if logo_public_id:
        try:
            await delete_logo(logo_public_id)
        except CloudinaryUploadError:
            logger.warning(
                "Failed to delete logo asset %s for template %s from Cloudinary "
                "— manual cleanup may be required",
                logo_public_id,
                id,
            )
        except Exception:
            logger.warning(
                "Failed to delete logo asset %s for template %s from Cloudinary "
                "— manual cleanup may be required",
                logo_public_id,
                id,
            )


async def edit_badge(
    session: AsyncSession,
    organiser_id: UUID,
    id: UUID,
    field_updates: dict[str, Any],
    new_hashtags: list[str] | None,
    update_hashtags: bool,
) -> Badge:
    result = await session.execute(
        select(Badge)
        .options(selectinload(Badge.hashtags))
        .where(
            Badge.id == id,
            Badge.deleted_at.is_(None),
        )
    )
    template = result.scalars().first()
    if template is None:
        raise BadgeNotFoundError
    if template.organiser_id != organiser_id:
        raise NotBadgeOwnerError

    for field, value in field_updates.items():
        setattr(template, field, value)

    if update_hashtags:
        template.hashtags.clear()
        for tag in new_hashtags or []:
            template.hashtags.append(BadgeHashtag(hashtag=tag))

    await session.commit()

    # Re-query after commit to return a fully consistent object with
    # the updated hashtag relationship loaded.
    refreshed = await session.execute(
        select(Badge).options(selectinload(Badge.hashtags)).where(Badge.id == id)
    )
    return refreshed.scalars().one()


async def get_badge_analytics(
    session: AsyncSession,
    organiser_id: UUID,
) -> tuple[int, int, int, int, list[tuple[UUID, int]]]:
    """Aggregate the authenticated organiser's badge metrics.

    Performs two database round-trips:
      1. A single SELECT that computes four scalar aggregates in one pass.
      2. A grouped SELECT for the per-template breakdown.

    Soft-deleted badges (``deleted_at IS NOT NULL``) are excluded from every
    aggregate to stay consistent with ``list_badges``.

    Returns:
        A tuple of (total, active, total_shares, total_creations, usage_rows)
        where ``usage_rows`` is a list of ``(platform_template_id, count)``
        ordered by count descending.
    """
    base_conditions = (
        Badge.organiser_id == organiser_id,
        Badge.deleted_at.is_(None),
    )

    scalar_stmt = select(
        func.count(Badge.id).label("total"),
        func.coalesce(
            func.sum(case((Badge.is_published.is_(True), 1), else_=0)), 0
        ).label("active"),
        func.coalesce(func.sum(Badge.share_count), 0).label("total_shares"),
        func.coalesce(func.sum(Badge.creation_count), 0).label("total_creations"),
    ).where(*base_conditions)

    scalar_result = await session.execute(scalar_stmt)
    total, active, total_shares, total_creations = scalar_result.one()

    usage_stmt = (
        select(
            Badge.platform_template_id,
            func.count(Badge.id).label("badge_count"),
        )
        .where(*base_conditions)
        .group_by(Badge.platform_template_id)
        .order_by(func.count(Badge.id).desc())
    )
    usage_rows = (await session.execute(usage_stmt)).all()

    return (
        int(total),
        int(active),
        int(total_shares),
        int(total_creations),
        [(row.platform_template_id, int(row.badge_count)) for row in usage_rows],
    )
