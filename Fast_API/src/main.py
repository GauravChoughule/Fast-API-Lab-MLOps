from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from predict import predict_data

app = FastAPI(title="Movie IMDb Rating Predictor API")


class MovieData(BaseModel):
    Production_Budget: float = Field(..., gt=0, description="Budget in US dollars")
    Running_Time_min: float = Field(..., gt=0, le=300, description="Runtime in minutes")
    Rotten_Tomatoes_Rating: float = Field(..., ge=0, le=100, description="Critics score, 0-100")
    Release_Year: int = Field(..., ge=1900, le=2030)
    Major_Genre: str = Field(..., description="e.g. Action, Comedy, Drama, Horror, Thriller/Suspense")
    MPAA_Rating: str = Field(..., description="e.g. G, PG, PG-13, R")
    Creative_Type: str = Field(..., description="e.g. Contemporary Fiction, Science Fiction, Fantasy")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "Production_Budget": 150000000, "Running_Time_min": 140,
                "Rotten_Tomatoes_Rating": 85, "Release_Year": 2008,
                "Major_Genre": "Action", "MPAA_Rating": "PG-13",
                "Creative_Type": "Super Hero",
            }]
        }
    }


class RatingResponse(BaseModel):
    predicted_imdb_rating: float
    verdict: str


def _verdict(rating: float) -> str:
    if rating >= 7.5:
        return "Great"
    if rating >= 6.5:
        return "Good"
    if rating >= 5.5:
        return "Average"
    return "Poor"


@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}


@app.post("/predict", response_model=RatingResponse)
async def predict_rating(movie: MovieData):
    try:
        rating = predict_data(movie.model_dump())
        return RatingResponse(predicted_imdb_rating=round(rating, 1), verdict=_verdict(rating))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))