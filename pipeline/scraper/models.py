from pydantic import BaseModel


class ImageResult(BaseModel):
    """Represents a discovered image result.

    Fields:
        title: Optional title or alt text
        url: URL of the image (remote)
        source_url: Optional page/source that hosts the image
        path: Optional local path if the scraper downloaded the image already
    """
    title: str | None = None
    url: str
    source_url: str | None = None
    path: str | None = None
