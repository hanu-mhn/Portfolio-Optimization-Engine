"""FastAPI server exposing optimization endpoints.

Endpoints:
- POST /load-data {tickers, start, end} -> returns data_id
- POST /optimize {method, data_id or tickers+start+end, other params} -> runs optimization and returns result
- GET /frontier?data_id=...&n_points=... -> returns MVO frontier points

This server uses an in-memory store `DATA_STORE` for loaded datasets. It's
intended as a lightweight dev server; for production you'd use persistent
storage and authentication.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid
import numpy as np

from .api import _fetch_prices_yfinance, run_optimization
from .data import compute_returns
from . import optimizer as opt_funcs

app = FastAPI()

# Simple in-memory store: data_id -> dict with 'prices', 'returns', 'mu', 'cov', 'tickers'
DATA_STORE: Dict[str, Dict[str, Any]] = {}


class LoadDataRequest(BaseModel):
    tickers: List[str]
    start: str
    end: str


class OptimizeRequest(BaseModel):
    method: str
    data_id: Optional[str] = None
    # allow passing tickers+start+end as alternatives
    tickers: Optional[List[str]] = None
    start: Optional[str] = None
    end: Optional[str] = None
    risk_free: Optional[float] = 0.0
    mar: Optional[float] = 0.0
    alpha: Optional[float] = 0.95
    benchmark: Optional[str] = None
    long_only: Optional[bool] = True
    min_weight: Optional[float] = None
    max_weight: Optional[float] = None
    n_frontier: Optional[int] = 50


@app.post('/load-data')
def load_data(req: LoadDataRequest):
    try:
        prices = _fetch_prices_yfinance(req.tickers, req.start, req.end)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    returns = compute_returns(prices, kind='simple')
    mu = returns.mean().values
    cov = returns.cov().values
    data_id = str(uuid.uuid4())
    DATA_STORE[data_id] = {'prices': prices, 'returns': returns.values, 'mu': mu, 'cov': cov, 'tickers': list(prices.columns)}
    return {'data_id': data_id, 'n_obs': len(returns), 'n_assets': len(prices.columns)}


@app.post('/optimize')
def optimize(req: OptimizeRequest):
    params = req.dict()
    # If data_id provided use that dataset
    if req.data_id:
        if req.data_id not in DATA_STORE:
            raise HTTPException(status_code=404, detail='data_id not found')
        d = DATA_STORE[req.data_id]
        prices_df = None
        # call run_optimization using in-memory prices
        res = run_optimization(method=req.method, prices=None, tickers=d['tickers'], start=None, end=None,
                               risk_free=req.risk_free, mar=req.mar, alpha=req.alpha, benchmark=req.benchmark,
                               long_only=req.long_only, min_weight=req.min_weight, max_weight=req.max_weight,
                               sum_to_one=True, n_frontier=req.n_frontier)
        # run_optimization expects to fetch prices if prices is None; instead pass prices DataFrame
        # to avoid fetching, call run_optimization with prices param
        res = run_optimization(method=req.method, prices=d['prices'], risk_free=req.risk_free, mar=req.mar, alpha=req.alpha, benchmark=req.benchmark, long_only=req.long_only, min_weight=req.min_weight, max_weight=req.max_weight, sum_to_one=True, n_frontier=req.n_frontier)
        return res

    # else require tickers+start+end
    if not (req.tickers and req.start and req.end):
        raise HTTPException(status_code=400, detail='Provide data_id or tickers+start+end')
    try:
        res = run_optimization(method=req.method, tickers=req.tickers, start=req.start, end=req.end, risk_free=req.risk_free, mar=req.mar, alpha=req.alpha, benchmark=req.benchmark, long_only=req.long_only, min_weight=req.min_weight, max_weight=req.max_weight, sum_to_one=True, n_frontier=req.n_frontier)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return res


@app.get('/frontier')
def get_frontier(data_id: str, n_points: int = 50):
    if data_id not in DATA_STORE:
        raise HTTPException(status_code=404, detail='data_id not found')
    d = DATA_STORE[data_id]
    mu = d['mu']
    cov = d['cov']
    tr, vols, wmat = opt_funcs.efficient_frontier(mu, cov, n_points=n_points)
    return {'target_returns': tr.tolist(), 'vols': vols.tolist(), 'weights_matrix': wmat.tolist()}
