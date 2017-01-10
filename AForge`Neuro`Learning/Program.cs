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
            Dictionary<string, double[]> last = opps.ToDictionary(k => k, v => GetVector(v, scores.Select(sc => sc.Value).ToArray()));

            // обучение

            List<List<double>> input = new List<List<double>> { };
            List<List<double>> output = new List<List<double>> { };

            scores = GetScores("NBA`11`12");

            foreach (var pair in scores.OrderBy(sc => sc.Key))
            {
                if (! opps.Contains(pair.Value.Split('#')[0].Trim())) continue;
                if (! opps.Contains(pair.Value.Split('#')[1].Trim())) continue;

                input.Add(new List<double> { });

                // данные за прошлый сезон команды дома
                input.Last().AddRange(last[pair.Value.Split('#')[0].Trim()]);

                // число дней с момента последней игры команды дома
                DateTime frDate = scores
                    .Where(sc => (new string[] { sc.Value.Split('#')[0].Trim(), sc.Value.Split('#')[1].Trim() }).Contains(pair.Value.Split('#')[0].Trim()))
                    .Min(sc => sc.Key);

                if (frDate == pair.Key)
                    input.Last().Add(7);
                else
                {
                    DateTime prDate = scores
                        .Where(sc => (new string[] { sc.Value.Split('#')[0].Trim(), sc.Value.Split('#')[1].Trim() }).Contains(pair.Value.Split('#')[0].Trim()))
                        .Where(sc => sc.Key < pair.Key).Max(sc => sc.Key);

                    input.Last().Add((pair.Key - prDate).TotalDays);
                }

                // данные за прошлый сезон команды в гостях
                input.Last().AddRange(last[pair.Value.Split('#')[1].Trim()]);

                // число дней с момента последней игры команды в гостях
                frDate = scores
                    .Where(sc => (new string[] { sc.Value.Split('#')[0].Trim(), sc.Value.Split('#')[1].Trim() }).Contains(pair.Value.Split('#')[1].Trim()))
                    .Min(sc => sc.Key);

                if (frDate == pair.Key)
                    input.Last().Add(7);
                else
                {
                    DateTime prDate = scores
                        .Where(sc => (new string[] { sc.Value.Split('#')[0].Trim(), sc.Value.Split('#')[1].Trim() }).Contains(pair.Value.Split('#')[1].Trim()))
                        .Where(sc => sc.Key < pair.Key).Max(sc => sc.Key);

                    input.Last().Add((pair.Key - prDate).TotalDays);
                }

                output.Add(Convert.ToInt32(pair.Value.Split('#')[2]) > Convert.ToInt32(pair.Value.Split('#')[3]) ? new List<double> { 1, 0 } : new List<double> { 0, 1 });
            }

            ActivationNetwork net = GetNetwork(input.Select(i => i.Select(ii => ii / 1E6).ToArray()).ToArray(), output.Select(o => o.Select(oo => oo / 1E6).ToArray()).ToArray());

            // анализ

            opps = scores.SelectMany(sc => new string[] { sc.Value.Split('#')[0].Trim(), sc.Value.Split('#')[1].Trim() }).Distinct().ToArray();
            last = opps.ToDictionary(k => k, v => GetVector(v, scores.Select(sc => sc.Value).ToArray()));

            scores = GetScores("NBA`12`13");

            int hit = 0;
            int cnt = 0;

            foreach (var pair in scores.OrderBy(sc => sc.Key))
            {
                if (! opps.Contains(pair.Value.Split('#')[0].Trim())) continue;
                if (! opps.Contains(pair.Value.Split('#')[1].Trim())) continue;

                List<double> list = new List<double> { };

                // данные за прошлый сезон команды дома
                list.AddRange(last[pair.Value.Split('#')[0].Trim()]);

                // число дней с момента последней игры команды дома
                DateTime frDate = scores
                    .Where(sc => (new string[] { sc.Value.Split('#')[0].Trim(), sc.Value.Split('#')[1].Trim() }).Contains(pair.Value.Split('#')[0].Trim()))
                    .Min(sc => sc.Key);

                if (frDate == pair.Key)
                    list.Add(7);
                else
                {
                    DateTime prDate = scores
                        .Where(sc => (new string[] { sc.Value.Split('#')[0].Trim(), sc.Value.Split('#')[1].Trim() }).Contains(pair.Value.Split('#')[0].Trim()))
                        .Where(sc => sc.Key < pair.Key).Max(sc => sc.Key);

                    list.Add((pair.Key - prDate).TotalDays);
                }

                // данные за прошлый сезон команды в гостях
                list.AddRange(last[pair.Value.Split('#')[1].Trim()]);

                // число дней с момента последней игры команды в гостях
                frDate = scores
                    .Where(sc => (new string[] { sc.Value.Split('#')[0].Trim(), sc.Value.Split('#')[1].Trim() }).Contains(pair.Value.Split('#')[1].Trim()))
                    .Min(sc => sc.Key);

                if (frDate == pair.Key)
                    list.Add(7);
                else
                {
                    DateTime prDate = scores
                        .Where(sc => (new string[] { sc.Value.Split('#')[0].Trim(), sc.Value.Split('#')[1].Trim() }).Contains(pair.Value.Split('#')[1].Trim()))
                        .Where(sc => sc.Key < pair.Key).Max(sc => sc.Key);

                    list.Add((pair.Key - prDate).TotalDays);
                }

                double[] result = net.Compute(list.ToArray());

                hit += (Convert.ToInt32(pair.Value.Split('#')[2]) > Convert.ToInt32(pair.Value.Split('#')[3]) && result[0] > result[1]) ? 1 : 0;
                hit += (Convert.ToInt32(pair.Value.Split('#')[2]) < Convert.ToInt32(pair.Value.Split('#')[3]) && result[0] < result[1]) ? 1 : 0;

                cnt++;
            }

            Console.WriteLine((double) hit / cnt);
            Console.ReadLine();
        }

        static ActivationNetwork GetNetwork(double[][] input, double[][] output)
        {
            ActivationNetwork net = new ActivationNetwork(new SigmoidFunction(), 18, 12, 8, 4, 2);
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

                    if (trainer.LearningRate < 0.001)
                        trainer.LearningRate = 0.001;
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

        static double[] GetVector(string opp, string[] scores)
        {
            double[] vector = new double[] {
                // число побед дома
                scores.Where(sc => sc.Split('#')[0].Trim() == opp).Count(sc => Convert.ToInt32(sc.Split('#')[2]) > Convert.ToInt32(sc.Split('#')[3])),
                // число побед в гостях
                scores.Where(sc => sc.Split('#')[1].Trim() == opp).Count(sc => Convert.ToInt32(sc.Split('#')[2]) < Convert.ToInt32(sc.Split('#')[3])),
                // число поражений дома
                scores.Where(sc => sc.Split('#')[0].Trim() == opp).Count(sc => Convert.ToInt32(sc.Split('#')[2]) < Convert.ToInt32(sc.Split('#')[3])),
                // число поражений в гостях
                scores.Where(sc => sc.Split('#')[1].Trim() == opp).Count(sc => Convert.ToInt32(sc.Split('#')[2]) > Convert.ToInt32(sc.Split('#')[3])),
                // число забитых мячей дома
                scores.Where(sc => sc.Split('#')[0].Trim() == opp).Sum(sc => Convert.ToInt32(sc.Split('#')[2])),
                // число забитых мячей в гостях
                scores.Where(sc => sc.Split('#')[1].Trim() == opp).Sum(sc => Convert.ToInt32(sc.Split('#')[3])),
                // число пропущенных мячей дома
                scores.Where(sc => sc.Split('#')[0].Trim() == opp).Sum(sc => Convert.ToInt32(sc.Split('#')[3])),
                // число пропущенных мячей в гостях
                scores.Where(sc => sc.Split('#')[1].Trim() == opp).Sum(sc => Convert.ToInt32(sc.Split('#')[2])),
            };

            return vector;
        }
    }
}
