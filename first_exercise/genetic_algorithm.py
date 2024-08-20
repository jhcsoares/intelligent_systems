import random
import math


class GeneticAlgorithm:
    population_size = None
    generations = None
    crossover_rate = None
    mutation_rate = None
    population_data = None
    victims_unified_map = None
    has_runned = None

    @classmethod
    def __generate_first_population(cls) -> None:

        for _ in range(0, cls.population_size):
            victims_id_list = random.sample(
                list(cls.victims_unified_map.keys()), len(cls.victims_unified_map)
            )
            random.shuffle(victims_id_list)
            cls.population_data["old_generation"]["population_list"].append(
                victims_id_list
            )

    @classmethod
    def __fitness_function(cls, victim_id: int) -> int:
        victim_data = cls.victims_unified_map[victim_id]
        coordinates = victim_data[0]

        euclidian_distance = math.sqrt(
            math.pow(coordinates[0], 2) + math.pow(coordinates[1], 2)
        )

        return 1 / euclidian_distance

    @classmethod
    def __population_rating(cls, population_data_dict: dict) -> None:
        fitness_results_list = []
        fitness_fix_factor = 0

        for i in range(0, cls.population_size):
            victims_sequence_list = population_data_dict["population_list"][i]

            evaluation = 0
            for j in range(0, len(victims_sequence_list)):
                victim_id = victims_sequence_list[j]
                evaluation += cls.__fitness_function(victim_id) * (j + 1)

            if evaluation < fitness_fix_factor:
                fitness_fix_factor = evaluation - 1

            fitness_results_list.append(evaluation)

        for i in range(0, cls.population_size):
            fitness_results_list[i] += abs(fitness_fix_factor)

        population_data_dict["fitness_results_list"] = fitness_results_list

        fitness_sum = sum(fitness_results_list)

        individual_probability_list = []
        for i in range(0, cls.population_size):
            evaluation = fitness_results_list[i] / fitness_sum
            individual_probability_list.append(evaluation)

        accumulated_probability_list = []
        accumulated_probability_sum = 0
        for i in range(0, cls.population_size):
            accumulated_probability_sum += individual_probability_list[i]
            accumulated_probability_list.append(accumulated_probability_sum)

        population_data_dict["accumulated_probability_list"] = (
            accumulated_probability_list
        )

    @classmethod
    def __breeding_individuals(cls, population_list, accumulated_probability_list):

        breedings_list = []
        for i in range(0, cls.population_size):
            choice = random.random() - 0.1
            for j in range(0, cls.population_size):
                if choice <= accumulated_probability_list[j]:
                    breedings_list.append(population_list[i])
                    break

        return breedings_list

    @classmethod
    def __crossover(cls, final_chromosome_breeding_1, final_chromosome_breeding_2):
        if random.random() <= cls.crossover_rate:
            crossover_slice_index = random.randint(0, len(final_chromosome_breeding_1))

            offspring_1 = final_chromosome_breeding_1[0:crossover_slice_index]
            offspring_2 = final_chromosome_breeding_1[
                crossover_slice_index : len(final_chromosome_breeding_1)
            ]

            for gene in final_chromosome_breeding_2:
                if gene not in offspring_1:
                    offspring_1.append(gene)
                if gene not in offspring_2:
                    offspring_2.append(gene)

        else:
            offspring_1 = final_chromosome_breeding_1
            offspring_2 = final_chromosome_breeding_2

        return (offspring_1, offspring_2)

    @classmethod
    def __mutation(cls, offspring_1: str, offspring_2: str):
        if random.random() <= cls.mutation_rate:
            first_allele_index = random.randint(0, len(offspring_1) - 1)
            second_allele_index = first_allele_index
            while first_allele_index == second_allele_index:
                second_allele_index = random.randint(0, len(offspring_1) - 1)

            gene_1 = offspring_1[first_allele_index]
            gene_2 = offspring_1[second_allele_index]
            offspring_1[first_allele_index] = gene_2
            offspring_1[second_allele_index] = gene_1

            first_allele_index = random.randint(0, len(offspring_2) - 1)
            second_allele_index = first_allele_index
            while first_allele_index == second_allele_index:
                second_allele_index = random.randint(0, len(offspring_2) - 1)

            gene_1 = offspring_2[first_allele_index]
            gene_2 = offspring_2[second_allele_index]
            offspring_2[first_allele_index] = gene_2
            offspring_2[second_allele_index] = gene_1

        return (offspring_1, offspring_2)

    @classmethod
    def __generate_offsprings(cls, breedings_list):

        offsprings_list = []

        while breedings_list:
            breeding_1 = random.choice(breedings_list)
            breedings_list.remove(breeding_1)

            breeding_2 = random.choice(breedings_list)
            breedings_list.remove(breeding_2)

            offspring_1, offspring_2 = cls.__crossover(breeding_1, breeding_2)

            offspring_1, offspring_2 = cls.__mutation(offspring_1, offspring_2)

            offsprings_list.append(offspring_1)
            offsprings_list.append(offspring_2)

        return offsprings_list

    @classmethod
    def __select_next_generation(cls) -> None:
        old_generation_data_list = cls.__combine_fitness_results_and_chromossomes(
            cls.population_data["old_generation"]
        )

        new_generation_data_list = cls.__combine_fitness_results_and_chromossomes(
            cls.population_data["new_generation"]
        )

        generations_data_list = old_generation_data_list + new_generation_data_list

        best_chromosomes_data_list = sorted(
            generations_data_list, key=lambda x: x[0], reverse=True
        )[0 : cls.population_size]

        for i in range(0, cls.population_size):
            cls.population_data["old_generation"]["population_list"][i] = (
                best_chromosomes_data_list[i][1]
            )

    @classmethod
    def __combine_fitness_results_and_chromossomes(cls, population_data_dict: dict):

        generation_data_list = []
        for i in range(0, cls.population_size):
            generation_data_list.append(
                (
                    population_data_dict["fitness_results_list"][i],
                    population_data_dict["population_list"][i],
                )
            )

        return generation_data_list

    @classmethod
    def __select_final_population(cls):
        fitness_results_list = cls.population_data["new_generation"][
            "fitness_results_list"
        ]
        best_fitness_value = 0
        best_fitness_value_index = -1

        for index in range(0, len(fitness_results_list)):
            if fitness_results_list[index] > best_fitness_value:
                best_fitness_value_index = fitness_results_list[index]
                best_fitness_value_index = index

        final_population = cls.population_data["new_generation"]["population_list"][
            best_fitness_value_index
        ]

        result = {0: [], 1: [], 2: [], 3: []}

        current_list_index = 0
        distribute_one_by_one = False

        for index in range(1, len(final_population) + 1):
            if not distribute_one_by_one:
                if index % (int(len(final_population) / 4)) == 0:
                    current_list_index += 1

                    if current_list_index == 4:
                        current_list_index = 0
                        distribute_one_by_one = True

                else:
                    result[current_list_index].append(final_population[index - 1])

            else:
                result[current_list_index].append(final_population[index - 1])
                current_list_index = (current_list_index + 1) % 4

        return result

    @classmethod
    def execute(
        cls,
        population_size: int,
        generations: int,
        crossover_rate: float,
        mutation_rate: float,
        victims_unified_map,
    ) -> None:

        cls.population_size = population_size
        cls.generations = generations
        cls.crossover_rate = crossover_rate
        cls.mutation_rate = mutation_rate
        cls.population_data = {
            "old_generation": {
                "population_list": [],
                "fitness_results_list": [],
                "accumulated_probability_list": [],
            },
            "new_generation": {
                "population_list": [],
                "fitness_results_list": [],
                "accumulated_probability_list": [],
            },
        }
        cls.victims_unified_map = victims_unified_map

        cls.__generate_first_population()

        for _ in range(0, cls.generations):
            cls.__population_rating(cls.population_data["old_generation"])

            breedings_list = cls.__breeding_individuals(
                population_list=cls.population_data["old_generation"][
                    "population_list"
                ],
                accumulated_probability_list=cls.population_data["old_generation"][
                    "accumulated_probability_list"
                ],
            )

            cls.population_data["new_generation"]["population_list"] = (
                cls.__generate_offsprings(breedings_list)
            )

            cls.__population_rating(cls.population_data["new_generation"])

            cls.__select_next_generation()

        return cls.__select_final_population()


if __name__ == "__main__":
    print(
        (
            GeneticAlgorithm.execute(
                population_size=16,
                generations=10000,
                crossover_rate=0.8,
                mutation_rate=0.04,
                victims_unified_map={
                    38: (
                        (-3, 8),
                        [38, 5.306324, 2.249959, -8.719746, 125.191957, 14.08686],
                    ),
                    39: ((0, 9), [39, 11.431293, 8.791381, -0.0, 132.188912, 9.174637]),
                    35: (
                        (8, 9),
                        [35, 14.716378, 8.118869, 1.70998, 60.346905, 10.946475],
                    ),
                    11: (
                        (9, 9),
                        [11, 12.523228, 11.83743, 0.367473, 45.521644, 8.15041],
                    ),
                    17: (
                        (11, 7),
                        [17, 20.212045, 6.0687, 4.765973, 116.558193, 10.77054],
                    ),
                    19: (
                        (6, 6),
                        [19, 18.187442, 5.487243, 4.773315, 175.802781, 2.707842],
                    ),
                    3: (
                        (8, 6),
                        [3, 13.53279, 8.097442, 0.733333, 66.812825, 18.422887],
                    ),
                    32: ((11, 6), [32, 12.735976, 6.2728, -0.0, 169.035637, 14.78786]),
                    2: (
                        (13, 5),
                        [2, 14.729957, 1.788218, -6.592956, 163.704849, 18.791823],
                    ),
                    24: (
                        (11, 4),
                        [24, 16.191573, 14.591113, 8.733333, 197.990041, 13.974354],
                    ),
                    5: (
                        (11, 3),
                        [5, 18.09571, 7.191948, 4.666667, 22.341149, 0.373625],
                    ),
                    14: (
                        (12, 1),
                        [14, 12.609438, 7.586974, 0.718312, 74.111905, 15.523695],
                    ),
                    21: (
                        (11, -1),
                        [21, 19.59081, 9.418446, 4.690964, 0.104475, 3.223328],
                    ),
                    33: (
                        (7, 3),
                        [33, 13.77814, 9.901845, 4.795896, 147.728058, 18.765959],
                    ),
                    4: (
                        (8, 3),
                        [4, 14.107792, 9.683278, 3.773222, 17.990136, 12.761988],
                    ),
                    25: (
                        (11, -3),
                        [25, 17.992675, 2.880425, -4.355487, 105.536014, 6.201071],
                    ),
                    37: (
                        (5, 3),
                        [37, 15.085302, 5.21569, 4.6954, 16.696563, 12.429551],
                    ),
                    13: (
                        (5, 2),
                        [13, 11.339571, 6.280149, 1.572637, 70.142535, 0.703802],
                    ),
                    16: (
                        (-1, 5),
                        [16, 10.002052, 0.953871, -4.788282, 139.33344, 1.089717],
                    ),
                    30: (
                        (-1, 4),
                        [30, 12.661543, 7.955163, 0.666667, 80.170498, 12.79403],
                    ),
                    9: (
                        (5, 1),
                        [9, 14.280553, 8.158289, 3.953607, 37.942081, 18.586825],
                    ),
                    15: (
                        (7, 0),
                        [15, 18.371217, 1.645462, -4.333333, 185.921773, 7.973052],
                    ),
                    41: (
                        (9, -2),
                        [41, 16.639092, 0.666811, -4.333333, 178.150423, 20.854529],
                    ),
                    28: (
                        (-5, -5),
                        [28, 15.230446, 1.4073, -4.343757, 45.568587, 14.980552],
                    ),
                    10: (
                        (-5, 3),
                        [10, 11.034661, 10.815699, 4.225599, 70.001165, 15.606912],
                    ),
                    6: (
                        (-3, 0),
                        [6, 14.488171, 9.589754, 4.166508, 27.25851, 2.658911],
                    ),
                    20: (
                        (-2, -1),
                        [20, 15.637701, 11.45257, 8.531563, 197.782323, 4.520872],
                    ),
                    27: (
                        (-5, -7),
                        [27, 14.229343, 10.443995, 4.568335, 160.269521, 15.293587],
                    ),
                    36: (
                        (-4, -7),
                        [36, 20.356882, 7.800787, 4.666667, 133.283243, 4.58615],
                    ),
                    29: (
                        (-1, -5),
                        [29, 13.074266, 7.881066, 0.737126, 60.618858, 14.787623],
                    ),
                    34: (
                        (0, -7),
                        [34, 21.87919, 10.071467, 4.824627, 117.197407, 19.226403],
                    ),
                    1: (
                        (2, -3),
                        [1, 18.954033, 4.771111, -6.834524, 157.992606, 19.91864],
                    ),
                    7: (
                        (13, -10),
                        [7, 17.621876, 8.170742, 4.666667, 135.730461, 18.979636],
                    ),
                    8: (
                        (9, -3),
                        [8, 14.362646, 9.709672, 1.728601, 99.035404, 10.654523],
                    ),
                    31: (
                        (8, -5),
                        [31, 14.900526, 7.917097, 0.733333, 74.932244, 12.72651],
                    ),
                    18: (
                        (8, -8),
                        [18, 12.700682, 6.725594, 1.738791, 163.079442, 4.235229],
                    ),
                    22: (
                        (6, -9),
                        [22, 19.447831, 11.579706, 8.554259, 173.087718, 4.159588],
                    ),
                    40: (
                        (5, -7),
                        [40, 11.284967, 3.93218, -4.598273, 145.950371, 4.531461],
                    ),
                    0: (
                        (2, -8),
                        [0, 12.137933, 6.323999, 0.615687, 70.460645, 16.012553],
                    ),
                    12: (
                        (2, -7),
                        [12, 14.915804, 8.905569, 0.733333, 67.994829, 13.857434],
                    ),
                    23: (
                        (3, -6),
                        [23, 17.014684, 13.992804, 8.733333, 122.513294, 0.938353],
                    ),
                },
            )
        )
    )
