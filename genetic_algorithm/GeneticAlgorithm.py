import random
import math


class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        generations: int,
        crossover_rate: float,
        mutation_rate: float,
        domain: list[int],
    ) -> None:
        self.__population_size = population_size
        self.__generations = generations
        self.__crossover_rate = crossover_rate
        self.__mutation_rate = mutation_rate
        self.__domain = domain
        self.__population_data = {
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

    def __generate_first_population(self) -> None:
        for _ in range(0, self.__population_size):
            self.__population_data["old_generation"]["population_list"].append(
                (
                    random.randint(self.__domain[0], self.__domain[1]),
                    random.randint(self.__domain[0], self.__domain[1]),
                )
            )

    def __fitness_function(self, gene_1: int, gene_2: int) -> int:
        return int(math.pow(gene_1, 3) - math.pow(gene_2, 3))

    def __population_rating(self, population_data_dict: dict) -> None:
        fitness_results_list = []
        fitness_fix_factor = 0

        for i in range(0, self.__population_size):
            gene_1 = population_data_dict["population_list"][i][0]
            gene_2 = population_data_dict["population_list"][i][1]

            evaluation = self.__fitness_function(gene_1, gene_2)

            if evaluation < fitness_fix_factor:
                fitness_fix_factor = evaluation - 1

            fitness_results_list.append(evaluation)

        for i in range(0, self.__population_size):
            fitness_results_list[i] += abs(fitness_fix_factor)

        for i in range(0, self.__population_size):
            if (
                population_data_dict["population_list"][i][0] < self.__domain[0]
                or population_data_dict["population_list"][i][1] < self.__domain[0]
                or population_data_dict["population_list"][i][0] > self.__domain[1]
                or population_data_dict["population_list"][i][1] > self.__domain[1]
            ):
                fitness_results_list[i] = 0

        population_data_dict["fitness_results_list"] = fitness_results_list

        fitness_sum = sum(fitness_results_list)

        individual_probability_list = []
        for i in range(0, self.__population_size):
            evaluation = fitness_results_list[i] / fitness_sum
            individual_probability_list.append(evaluation)

        accumulated_probability_list = []
        accumulated_probability_sum = 0
        for i in range(0, self.__population_size):
            accumulated_probability_sum += individual_probability_list[i]
            accumulated_probability_list.append(accumulated_probability_sum)

        population_data_dict["accumulated_probability_list"] = (
            accumulated_probability_list
        )

    def __breeding_individuals(
        self, population_list: list[(int, int)], accumulated_probability_list: list[int]
    ) -> list[(int, int)]:

        breedings_list = []
        for i in range(0, self.__population_size):
            choice = random.random() - 0.1
            for j in range(0, self.__population_size):
                if choice <= accumulated_probability_list[j]:
                    breedings_list.append(population_list[i])
                    break

        return breedings_list

    def __crossover(
        self, final_chromosome_breeding_1: str, final_chromosome_breeding_2: str
    ) -> tuple[str, str]:
        if random.random() <= self.__crossover_rate:
            crossover_slice_index = random.randint(0, len(final_chromosome_breeding_1))

            offspring_1 = (
                final_chromosome_breeding_1[0:crossover_slice_index]
                + final_chromosome_breeding_2[
                    crossover_slice_index : len(final_chromosome_breeding_1)
                ]
            )

            offspring_2 = (
                final_chromosome_breeding_2[0:crossover_slice_index]
                + final_chromosome_breeding_1[
                    crossover_slice_index : len(final_chromosome_breeding_1)
                ]
            )

        else:
            offspring_1 = final_chromosome_breeding_1
            offspring_2 = final_chromosome_breeding_2

        return (offspring_1, offspring_2)

    def __mutation(self, offspring_1: str, offspring_2: str) -> tuple[str, str]:
        if random.random() <= self.__mutation_rate:
            number_of_mutated_alleles_1 = random.randint(1, len(offspring_1))

            mutated_alleles_indexes = set()
            for _ in range(0, number_of_mutated_alleles_1):
                mutated_alleles_indexes.add(random.randint(0, len(offspring_1) - 1))

            offspring_1 = list(offspring_1)

            for index in mutated_alleles_indexes:
                allele = offspring_1[index]

                if allele == "0":
                    offspring_1[index] = "1"
                else:
                    offspring_1[index] = "0"

            offspring_1 = "".join(offspring_1)

            number_of_mutated_alleles_2 = random.randint(1, len(offspring_2))

            mutated_alleles_indexes = set()
            for _ in range(0, number_of_mutated_alleles_2):
                mutated_alleles_indexes.add(random.randint(0, len(offspring_2) - 1))

            offspring_2 = list(offspring_2)

            for index in mutated_alleles_indexes:
                allele = offspring_2[index]

                if allele == "0":
                    offspring_2[index] = "1"
                else:
                    offspring_2[index] = "0"

            offspring_2 = "".join(offspring_2)

        return (offspring_1, offspring_2)

    def __generate_offsprings(
        self, breedings_list: list[(int, int)]
    ) -> list[(int, int)]:

        offsprings_list = []

        while breedings_list:
            breeding_1 = random.choice(breedings_list)
            breedings_list.remove(breeding_1)

            binary_format_string = "0" + str(len(bin(self.__domain[1])[2:])) + "b"

            gene_1_breeding_1 = format(breeding_1[0], binary_format_string)
            gene_2_breeding_1 = format(breeding_1[1], binary_format_string)
            final_chromosome_breeding_1 = gene_1_breeding_1 + gene_2_breeding_1

            breeding_2 = random.choice(breedings_list)
            breedings_list.remove(breeding_2)

            gene_1_breeding_2 = format(breeding_2[0], binary_format_string)
            gene_2_breeding_2 = format(breeding_2[1], binary_format_string)
            final_chromosome_breeding_2 = gene_1_breeding_2 + gene_2_breeding_2

            offspring_1, offspring_2 = self.__crossover(
                final_chromosome_breeding_1, final_chromosome_breeding_2
            )

            offspring_1, offspring_2 = self.__mutation(offspring_1, offspring_2)

            offsprings_list.append(
                (
                    int(offspring_1[: len(gene_1_breeding_1)], 2),
                    int(offspring_1[len(gene_1_breeding_1) :], 2),
                )
            )
            offsprings_list.append(
                (
                    int(offspring_2[: len(gene_1_breeding_2)], 2),
                    int(offspring_2[len(gene_1_breeding_1) :], 2),
                )
            )

        return offsprings_list

    def __select_next_generation(self) -> None:
        old_generation_data_list = self.__combine_fitness_results_and_chromossomes(
            self.__population_data["old_generation"]
        )

        new_generation_data_list = self.__combine_fitness_results_and_chromossomes(
            self.__population_data["new_generation"]
        )

        generations_data_list = old_generation_data_list + new_generation_data_list

        best_chromosomes_data_list = sorted(
            generations_data_list, key=lambda x: x[0], reverse=True
        )[0 : self.__population_size]

        for i in range(0, self.__population_size):
            self.__population_data["old_generation"]["population_list"][i] = (
                best_chromosomes_data_list[i][1]
            )

    def __combine_fitness_results_and_chromossomes(
        self, population_data_dict: dict
    ) -> list[int, (int, int)]:

        generation_data_list = []
        for i in range(0, self.__population_size):
            generation_data_list.append(
                (
                    population_data_dict["fitness_results_list"][i],
                    population_data_dict["population_list"][i],
                )
            )

        return generation_data_list

    def execute(self) -> None:
        self.__generate_first_population()

        for _ in range(0, self.__generations):
            self.__population_rating(self.__population_data["old_generation"])

            breedings_list = self.__breeding_individuals(
                population_list=self.__population_data["old_generation"][
                    "population_list"
                ],
                accumulated_probability_list=self.__population_data["old_generation"][
                    "accumulated_probability_list"
                ],
            )

            self.__population_data["new_generation"]["population_list"] = (
                self.__generate_offsprings(breedings_list)
            )

            self.__population_rating(self.__population_data["new_generation"])

            self.__select_next_generation()

        print(self.__population_data["new_generation"]["population_list"])


ga = GeneticAlgorithm(
    population_size=32,
    generations=40000,
    crossover_rate=0.8,
    mutation_rate=0.05,
    domain=[4, 273],
)

ga.execute()
