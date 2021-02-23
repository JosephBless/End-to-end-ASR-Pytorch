import re

def default_cost(alignment_matrix, i, j, tokens_1, tokens_2):
    """
    Ignore input and return 1 for all cases (insertion, deletion, substitution)
    """
    return 1


class UtteranceAligner:
    """
    A utility class that align two utterances with Levenshtein alignment and customized cost
    """
    INSERTION_SYM = 'I'
    DELETION_SYM = 'D'
    SUBSTITUTION_SYM = 'S'
    MATCH_SYM = '-'

    def __init__(self,
                 substitution_cost=default_cost,
                 insertion_cost=default_cost,
                 deletion_cost=default_cost):
        self.substitution_cost = substitution_cost
        self.insertion_cost = insertion_cost
        self.deletion_cost = deletion_cost


    def align_utterances(self, utterance_1, utterance_2):
        """
        Align utterance_2 to utterance_1. A token in utterance_1 but not in utterance_2 is a deletion
        and a token in utterance_2 but not in utterance_1 is an insertion.
        """
        # TODO: Integrate locale-specific tokenizers
        tokens_1 = re.split('\s+', utterance_1)
        tokens_2 = re.split('\s+', utterance_2)

        n_tokens_1 = len(tokens_1)
        n_tokens_2 = len(tokens_2)

        # in each element of the alignment_matrix, the first value is the alignment cost and
        # the second is alignment trace
        alignment_matrix = [
            [(0, '') for j in range(0, n_tokens_2+1)] for i in range(0, n_tokens_1+1)]

        for i in range(1, n_tokens_1+1):
            cost_payload = (alignment_matrix, i, 0, tokens_1, tokens_2)
            alignment_matrix[i][0] = (
                alignment_matrix[i-1][0][0] + self.deletion_cost(*cost_payload),
                alignment_matrix[i-1][0][1] + self.DELETION_SYM)

        for j in range(1, n_tokens_2+1):
            cost_payload = (alignment_matrix, 0, j, tokens_1, tokens_2)
            alignment_matrix[0][j] = (
                alignment_matrix[0][j-1][0] + self.insertion_cost(*cost_payload),
                alignment_matrix[0][j-1][1] + self.INSERTION_SYM)

        for i in range(1, n_tokens_1+1):
            for j in range(1, n_tokens_2+1):
                if tokens_1[i-1] == tokens_2[j-1]:
                    alignment_matrix[i][j] = (
                        alignment_matrix[i-1][j-1][0], alignment_matrix[i-1][j-1][1] + self.MATCH_SYM)
                else:
                    cost_payload = (alignment_matrix, i, j, tokens_1, tokens_2)
                    alignment_matrix[i][j] = min(
                        (alignment_matrix[i-1][j][0] + self.deletion_cost(*cost_payload),
                            alignment_matrix[i-1][j][1] + self.DELETION_SYM),
                        (alignment_matrix[i][j-1][0] + self.insertion_cost(*cost_payload),
                            alignment_matrix[i][j-1][1] + self.INSERTION_SYM),
                        (alignment_matrix[i-1][j-1][0] + self.substitution_cost(*cost_payload),
                            alignment_matrix[i-1][j-1][1] + self.SUBSTITUTION_SYM)
                    )

        return alignment_matrix[-1][-1]
