using Microsoft.Extensions.ML;
using Microsoft.ML.Data;
using Microsoft.AspNetCore.Http;
using ImageClassficationCoin;

namespace TestAIDog
{
    public class PredictionResult
    {
        public string[] PredictedLabel { get; set; }
        public float[] Score { get; set; }
    }

    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Register PredictionEnginePool service
            builder.Services.AddPredictionEnginePool<MLModel.ModelInput, MLModel.ModelOutput>()
                .FromFile("MLModel.mlnet");

            builder.Services.AddControllers();
            builder.Services.AddEndpointsApiExplorer();
            builder.Services.AddSwaggerGen();

            var app = builder.Build();

            if (app.Environment.IsDevelopment())
            {
                app.UseSwagger();
                app.UseSwaggerUI();
            }

            string uploadFolder = Path.Combine(Directory.GetCurrentDirectory(), "uploads");
            Directory.CreateDirectory(uploadFolder);

            app.MapPost("/predict", async (
     HttpRequest request,
     PredictionEnginePool<MLModel.ModelInput, MLModel.ModelOutput> predictionEnginePool) =>
            {
                if (!request.HasFormContentType || request.Form.Files.Count == 0)
                {
                    return Results.BadRequest("Please upload an image file.");
                }

                IFormFile file = request.Form.Files[0];

                using var memoryStream = new MemoryStream();
                await file.CopyToAsync(memoryStream);
                byte[] imageBytes = memoryStream.ToArray();

                var input = new MLModel.ModelInput()
                {
                    ImageSource = imageBytes,
                };

                var prediction = predictionEnginePool.Predict(input);

                var result = new PredictionResult
                {
                    PredictedLabel = new[] { prediction.PredictedLabel },
                    Score = prediction.Score
                };

                return Results.Ok(result);
            });


            app.UseHttpsRedirection();
            app.UseAuthorization();
            app.MapControllers();
            app.Run();
        }
    }
}