"""
Cloud Run Job Worker - Runs RL training jobs.

This script runs as a Cloud Run Job and executes the actual training.
Progress updates are sent to Firestore for the UI to display.
"""

import os
import sys
import logging
import time
import tempfile
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import stockpatternannotator
sys.path.insert(0, str(Path(__file__).parent.parent))

from stockpatternannotator import RLPipeline, PolygonClient, DatabaseManager
from webapp.firestore_client import FirestoreClient
from webapp.storage_client import StorageClient
from webapp.models import JobStatus

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingWorker:
    """Worker that executes training jobs."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.firestore = FirestoreClient()
        self.storage = StorageClient()

        # Get job config from environment
        self.polygon_api_key = os.getenv("POLYGON_API_KEY")
        self.symbol = os.getenv("SYMBOL")
        self.timeframe = os.getenv("TIMEFRAME")
        self.start_date = os.getenv("START_DATE")
        self.end_date = os.getenv("END_DATE")
        self.total_timesteps = int(os.getenv("TOTAL_TIMESTEPS", "100000"))
        self.forecast_horizon = int(os.getenv("FORECAST_HORIZON", "5"))
        self.initial_balance = float(os.getenv("INITIAL_BALANCE", "10000.0"))
        self.transaction_cost = float(os.getenv("TRANSACTION_COST", "0.001"))
        self.allow_short = os.getenv("ALLOW_SHORT", "false").lower() == "true"

        # Create temp directory for local files
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "stockpatterns.db")

    def update_progress(self, progress: float, message: str):
        """Update job progress in Firestore."""
        logger.info(f"Progress: {progress:.1f}% - {message}")
        self.firestore.update_job_progress(self.job_id, progress, message)

    def update_status(self, status: JobStatus, message: str = None):
        """Update job status in Firestore."""
        logger.info(f"Status: {status.value} - {message or ''}")
        self.firestore.update_job_status(self.job_id, status, message)

    def run(self):
        """Execute the training job."""
        start_time = time.time()

        try:
            # Update status to running
            self.update_status(JobStatus.RUNNING, "Initializing...")
            self.update_progress(0, "Setting up environment")

            # Step 1: Fetch data from Polygon.io
            self.update_progress(10, f"Fetching {self.symbol} data from Polygon.io")
            logger.info(f"Fetching {self.symbol} {self.timeframe} from {self.start_date} to {self.end_date}")

            polygon_client = PolygonClient(api_key=self.polygon_api_key)
            ohlc_data = polygon_client.get_aggregates(
                ticker=self.symbol,
                timespan=self.timeframe[:-1],  # Remove 'M', 'H', 'D'
                from_date=self.start_date,
                to_date=self.end_date,
                multiplier=1
            )

            if ohlc_data.empty:
                raise ValueError(f"No data returned for {self.symbol}")

            logger.info(f"Fetched {len(ohlc_data)} candles")

            # Step 2: Store in local database
            self.update_progress(20, "Storing data in database")

            db = DatabaseManager(f"sqlite:///{self.db_path}")
            db.save_ohlc_data(ohlc_data, self.symbol, self.timeframe)

            # Step 3: Initialize RL Pipeline
            self.update_progress(30, "Initializing RL pipeline")

            pipeline = RLPipeline(
                database_url=f"sqlite:///{self.db_path}",
                forecast_horizon=self.forecast_horizon,
                test_size=0.2,
                random_state=42,
                show_gpu_info=False  # Don't print GPU info in Cloud Run
            )

            # Step 4: Load and validate patterns
            self.update_progress(40, "Loading data and validating patterns")

            pipeline.load_data(symbol=self.symbol, timeframe=self.timeframe)
            pipeline.validate_patterns()

            # Step 5: Calculate features
            self.update_progress(50, "Calculating technical indicators")

            pipeline.calculate_features()

            # Step 6: Prepare environments
            self.update_progress(60, "Preparing training environment")

            env_config = {
                'initial_balance': self.initial_balance,
                'transaction_cost': self.transaction_cost,
                'allow_short': self.allow_short,
                'max_position_size': 1.0,
                'max_drawdown': 0.2,
            }

            pipeline.prepare_environments(env_config)

            # Step 7: Train agent
            self.update_progress(70, f"Training agent ({self.total_timesteps:,} timesteps)")

            logger.info(f"Starting training for {self.total_timesteps} timesteps")

            model_path = os.path.join(self.temp_dir, "model.zip")

            agent = pipeline.train_agent(
                total_timesteps=self.total_timesteps,
                hyperparameters=None,  # Use auto-optimized
                save_path=model_path
            )

            # Step 8: Evaluate
            self.update_progress(85, "Evaluating agent")

            eval_results = pipeline.evaluate_agent(n_episodes=10)

            logger.info(f"Evaluation: {eval_results}")

            # Step 9: Backtest
            self.update_progress(90, "Running backtest")

            backtest_df = pipeline.backtest()
            backtest_path = os.path.join(self.temp_dir, "backtest.csv")
            backtest_df.to_csv(backtest_path, index=False)

            # Step 10: Upload results
            self.update_progress(95, "Uploading results")

            # Upload model
            model_gcs_path = self.storage.upload_model(self.job_id, model_path)

            # Upload backtest CSV
            csv_gcs_path = self.storage.upload_backtest_csv(self.job_id, backtest_path)

            # Calculate metrics
            final_balance = backtest_df.iloc[-1]['balance']
            total_return_pct = (final_balance - self.initial_balance) / self.initial_balance * 100

            metrics = {
                'mean_return': eval_results['mean_return'],
                'std_return': eval_results['std_return'],
                'mean_win_rate': eval_results['mean_win_rate'],
                'mean_trades': eval_results['mean_trades'],
                'final_balance': final_balance,
                'total_return_pct': total_return_pct,
                'total_trades': int(eval_results['mean_trades']),
                'data_candles': len(ohlc_data),
                'training_duration_seconds': time.time() - start_time
            }

            # Upload results JSON
            results_gcs_path = self.storage.upload_results(self.job_id, {
                'job_id': self.job_id,
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'metrics': metrics,
                'backtest_csv': csv_gcs_path,
                'model': model_gcs_path,
                'completed_at': datetime.utcnow().isoformat()
            })

            # Update job with results
            self.firestore.update_job_results(
                self.job_id,
                model_path=model_gcs_path,
                results_path=results_gcs_path,
                metrics=metrics
            )

            self.update_progress(100, "Training completed successfully")
            self.update_status(JobStatus.COMPLETED, f"Completed in {time.time() - start_time:.0f}s")

            logger.info(f"Job {self.job_id} completed successfully")
            logger.info(f"Final balance: ${final_balance:,.2f} ({total_return_pct:+.2f}%)")

        except Exception as e:
            logger.error(f"Job failed: {e}", exc_info=True)
            self.update_status(JobStatus.FAILED, str(e))
            raise

        finally:
            # Cleanup temp directory
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir: {e}")


def main():
    """Main entry point."""
    job_id = os.getenv("JOB_ID")

    if not job_id:
        logger.error("JOB_ID environment variable not set")
        sys.exit(1)

    logger.info(f"Starting worker for job {job_id}")

    worker = TrainingWorker(job_id)
    worker.run()

    logger.info("Worker finished successfully")


if __name__ == "__main__":
    main()
