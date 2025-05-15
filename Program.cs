using Microsoft.Extensions.ML;
using Microsoft.ML.Data;
using Microsoft.AspNetCore.Http;
using ImageClassficationCoin;

namespace TestAIDog
{
    public class PredictionResponse
    {
        public string Status { get; set; }
        public string Message { get; set; }
        public object Data { get; set; }
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
                    return Results.BadRequest(new PredictionResponse
                    {
                        Status = "error",
                        Message = "Please upload an image file.",
                        Data = null
                    });
                }

                IFormFile file = request.Form.Files[0];

                using var memoryStream = new MemoryStream();
                await file.CopyToAsync(memoryStream);
                byte[] imageBytes = memoryStream.ToArray();

                var input = new MLModel.ModelInput()
                {
                    ImageSource = imageBytes,
                };

                try
                {
                    var prediction = predictionEnginePool.Predict(input);

                    // Check if the predicted label indicates a coin
                    bool isCoin = prediction.PredictedLabel == "ValidCoin";

                    var result = new PredictionResponse
                    {
                        Status = "success",
                        Message = "Prediction completed",
                        Data = new { isCoin }
                    };

                    return Results.Ok(result);
                }
                catch (Exception ex)
                {
                    return Results.Problem(new PredictionResponse
                    {
                        Status = "error",
                        Message = $"Prediction failed: {ex.Message}",
                        Data = null
                    }.ToString());
                }
            });

            app.UseHttpsRedirection();
            app.UseAuthorization();
            app.MapControllers();
            app.Run();
        }
    }
}
