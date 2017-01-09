using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using AngleSharp.Dom.Html;
using AngleSharp.Parser.Html;
using AForge.Neuro;
using AForge.Neuro.Learning;

namespace AForge_Neuro_Learning
{
    class Program
    {
        static void Main()
        {
            Dictionary<DateTime, string> scores = GetScores("NBA`10`11");

            string[] opps = scores.SelectMany(sc => new string[] { sc.Value.Split('#')[0].Trim(), sc.Value.Split('#')[1].Trim() }).Distinct().ToArray();

            Dictionary<string, double[]> last = opps.ToDictionary(
                k => k,
                v => new double[] {
                    // число побед дома
                    scores.Values.Where(sc => sc.Split('#')[0].Trim() == v).Count(sc => Convert.ToInt32(sc.Split('#')[2]) > Convert.ToInt32(sc.Split('#')[3])),
                    // число побед в гостях
                    scores.Values.Where(sc => sc.Split('#')[1].Trim() == v).Count(sc => Convert.ToInt32(sc.Split('#')[2]) < Convert.ToInt32(sc.Split('#')[3])),
                    // число поражений дома
                    scores.Values.Where(sc => sc.Split('#')[0].Trim() == v).Count(sc => Convert.ToInt32(sc.Split('#')[2]) < Convert.ToInt32(sc.Split('#')[3])),
                    // число поражений в гостях
                    scores.Values.Where(sc => sc.Split('#')[1].Trim() == v).Count(sc => Convert.ToInt32(sc.Split('#')[2]) > Convert.ToInt32(sc.Split('#')[3])),
                    // число забитых мячей дома
                    scores.Values.Where(sc => sc.Split('#')[0].Trim() == v).Sum(sc => Convert.ToInt32(sc.Split('#')[2])),
                    // число забитых мячей в гостях
                    scores.Values.Where(sc => sc.Split('#')[1].Trim() == v).Sum(sc => Convert.ToInt32(sc.Split('#')[3])),
                    // число пропущенных мячей дома
                    scores.Values.Where(sc => sc.Split('#')[0].Trim() == v).Sum(sc => Convert.ToInt32(sc.Split('#')[3])),
                    // число пропущенных мячей в гостях
                    scores.Values.Where(sc => sc.Split('#')[1].Trim() == v).Sum(sc => Convert.ToInt32(sc.Split('#')[2])),
                });

            List<List<double>> input = new List<List<double>> { };
            List<List<double>> output = new List<List<double>> { };

            scores = GetScores("NBA`11`12");

            foreach (var pair in scores.OrderBy(pair => pair.Key))
            {
                if (! opps.Contains(pair.Value.Split('#')[0].Trim())) continue; 
                if (! opps.Contains(pair.Value.Split('#')[1].Trim())) continue;

                input.Add(new List<double> { });

                // данные за прошлый сезон команды дома
                input.Last().AddRange(last[pair.Value.Split('#')[0].Trim()]);
                // данные за прошлый сезон команды в гостях
                input.Last().AddRange(last[pair.Value.Split('#')[1].Trim()]);

                output.Add(new List<double> { });

                if (Convert.ToInt32(pair.Value.Split('#')[2]) > Convert.ToInt32(pair.Value.Split('#')[3]))
                    output.Last().AddRange(new List<double> { 1, 0 });
                else
                    output.Last().AddRange(new List<double> { 0, 1 });
            }

            ActivationNetwork net = GetNetwork(input.Select(i => i.ToArray()).ToArray(), output.Select(o => o.ToArray()).ToArray());

            //double[] result = net.Compute(new double[] { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 });
        }

        static ActivationNetwork GetNetwork(double[][] input, double[][] output)
        {
            ActivationNetwork net = new ActivationNetwork(new SigmoidFunction(), 16, 24, 32, 24, 16, 12, 8, 4, 2);
            BackPropagationLearning trainer = new BackPropagationLearning(net);

            // Переменная, сохраняющая значение ошибки сети на предыдущем шаге
            double early = 1E6;
            // Ошибка сети
            double error = 1E3;
            // Сначала скорость обучения должна быть высока
            trainer.LearningRate = 1;
            // Обучаем сеть пока ошибка сети станет небольшой
            while (error > 0.01)
            {
                // Получаем ошибку сети
                error = trainer.RunEpoch(input, output);

                // Если ошибка сети изменилась на небольшое значения, в сравнении ошибкой предыдущей эпохи
                if (Math.Abs(error - early) < 1E-9)
                {
                    // Уменьшаем коэффициент скорости обучения на 2
                    trainer.LearningRate /= 2;

                    if (trainer.LearningRate < 0.01)
                        trainer.LearningRate = 0.01;
                }

                Console.WriteLine(error + " / " + trainer.LearningRate);
                early = error;
            }

            return net;
        }

        static Dictionary<DateTime, string> GetScores(string mark)
        {
            Dictionary<DateTime, string> dictionary = new Dictionary<DateTime, string> { };

            string html = string.Join("", File.ReadAllLines(Directory.GetCurrentDirectory() + @"\DATA\" + mark + ".txt"));
            IHtmlDocument doc = new HtmlParser().Parse(html);

            var body = doc.QuerySelector("body");
            var page = body.QuerySelectorAll("div").ToList().Find(x => x.GetAttribute("class") == "page");
            var table = page.QuerySelectorAll("table").ToList().Find(x => x.GetAttribute("class") == "table b-table-sortlist");
            var tbody = table.QuerySelector("tbody");

            foreach (var tr in tbody.QuerySelectorAll("tr"))
            {
                var dt = tr.QuerySelectorAll("td").ToList().Find(x => x.GetAttribute("class") == "sport__calendar__table__date");
                DateTime date = DateTime.Parse(dt.TextContent.Replace(", пер.", ""));
                var names = tr.QuerySelectorAll("a").ToList().FindAll(x => x.GetAttribute("class") == "sport__calendar__table__team").Select(n => n.TextContent.Trim());
                var one = tr.QuerySelectorAll("span").ToList().Find(x => x.GetAttribute("class") == "sport__calendar__table__result__left").TextContent.Trim();
                var two = tr.QuerySelectorAll("span").ToList().Find(x => x.GetAttribute("class") == "sport__calendar__table__result__right").TextContent.Trim();

                while (dictionary.ContainsKey(date))
                    date = date.AddMinutes(1);

                dictionary.Add(date, string.Join(" # ", names) + " # " + one + " # " + two);
            }

            return dictionary;
        }
    }
}
