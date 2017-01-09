using System;
using AForge.Neuro;
using AForge.Neuro.Learning;

namespace AForge_Neuro_Learning
{
    class Program
    {
        static void Main()
        {
            // Создаем сеть с сигмодиной активацонной функцией и 4 слоями (с 10,3,3,10 нейронами в каждом слое, соответственно)
            ActivationNetwork net = new ActivationNetwork(new SigmoidFunction(), 10, 3, 3, 10);
            // Обучение - алгоритм с обратным распространением ошибки
            BackPropagationLearning trainer = new BackPropagationLearning(net);

            // Формируем множество входных векторов
            double[][] input = new double[][] {
                new double[]{1,1,1,1,1,1,1,1,1,1},
                new double[]{0,0,0,0,0,0,0,0,0,0},
                new double[]{1,1,1,1,1,0,0,0,0,0},
                new double[]{0,0,0,0,0,1,1,1,1,1},
            };

            // Формируем множество желаемых выходных векторов
            double[][] output = new double[][] {
                new double[]{1,1,1,1,1,1,1,1,1,1},
                new double[]{0,0,0,0,0,0,0,0,0,0},
                new double[]{1,1,1,1,1,0,0,0,0,0},
                new double[]{0,0,0,0,0,1,1,1,1,1},
            };

            // Переменная, сохраняющая значение ошибки сети на предыдущем шаге
            double prErr = 10000000;
            // Ошибка сети
            double error = 100;
            // Сначала скорость обучения должна быть высока
            trainer.LearningRate = 1;
            // Обучаем сеть пока ошибка сети станет небольшой
            while (error > 0.001)
            {
                // Получаем ошибку сети
                error = trainer.RunEpoch(input, output);
                // Если ошибка сети изменилась на небольшое значения, в сравнении ошибкой предыдущей эпохи
                if (Math.Abs(error - prErr) < 0.000000001)
                {
                    // Уменьшаем коэффициент скорости обучения на 2
                    trainer.LearningRate /= 2;
                    if (trainer.LearningRate < 0.001)
                        trainer.LearningRate = 0.001;
                }

                prErr = error;
            }

            double[] result;
            result = net.Compute(new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 });
            result = net.Compute(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
            result = net.Compute(new double[] { 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 });
            result = net.Compute(new double[] { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 });
        }
    }
}
