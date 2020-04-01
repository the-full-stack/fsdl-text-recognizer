"""Script to output multiple Markdown files by substituting part of the template with paragraphs of text."""
from jinja2 import Template


PARAGRAPHS = [
    (
        "Deep learning (also known as deep structured learning or hierarchical learning) is part of a broader family of machine learning methods based on learning data representations, as opposed to task-specific algorithms. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, deep belief networks and recurrent neural networks have been applied to fields including computer vision, speech recognition, natural language processing, and audio recognition.",
        "https://en.wikipedia.org/wiki/Deep_learning",
    ),
    (
        "Gradient descent is a first-order iterative optimization algorithm for finding the minimum of a function. To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient (or approximate gradient) of the function at the current point. If, instead, one takes steps proportional to the positive of the gradient, one approaches a local maximum of that function; the procedure is then known as gradient ascent.",
        "https://en.wikipedia.org/wiki/Gradient_descent",
    ),
    (
        "In mathematical analysis, the maxima and minima (the respective plurals of maximum and minimum) of a function, known collectively as extrema (the plural of extremum), are the largest and smallest value of the function, either within a given range (the local or relative extrema) or on the entire domain of a function (the global or absolute extrema). Pierre de Fermat was one of the first mathematicians to propose a general technique, adequality, for finding the maxima and minima of functions.",
        "https://en.wikipedia.org/wiki/Maxima_and_minima",
    ),
    (
        "Mathematical analysis is the branch of mathematics dealing with limits and related theories, such as differentiation, integration, measure, infinite series, and analytic functions. These theories are usually studied in the context of real and complex numbers and functions. Analysis evolved from calculus, which involves the elementary concepts and techniques of analysis.",
        "https://en.wikipedia.org/wiki/Mathematical_analysis",
    ),
    (
        "Mathematicians seek and use patterns to formulate new conjectures; they resolve the truth or falsity of conjectures by mathematical proof. When mathematical structures are good models of real phenomena, then mathematical reasoning can provide insight or predictions about nature. Through the use of abstraction and logic, mathematics developed from counting, calculation, measurement, and the systematic study of the shapes and motions of physical objects.",
        "https://en.wikipedia.org/wiki/Mathematics",
    ),
    (
        "Natural language processing (NLP) is a subfield of computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data. Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation.",
        "https://en.wikipedia.org/wiki/Natural_language_processing",
    ),
    (
        "From the technology perspective, speech recognition has a long history with several waves of major innovations. Most recently, the field has benefited from advances in deep learning and big data. The advances are evidenced not only by the surge of academic papers published in the field, but more importantly by the worldwide industry adoption of a variety of deep learning methods in designing and deploying speech recognition systems.",
        "https://en.wikipedia.org/wiki/Speech_recognition",
    ),
    (
        "Borrowing from the management literature, Kaplan and Haenlein classify artificial intelligence into three different types of AI systems: analytical, human-inspired, and humanized artificial intelligence. Analytical AI has only characteristics consistent with cognitive intelligence; generating a cognitive representation of the world and using learning based on past experience to inform future decisions. Human-inspired AI has elements from cognitive and emotional intelligence; understanding human emotions, in addition to cognitive elements, and considering them in their decision making.",
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
    ),
    (
        "Computer science is the study of processes that interact with data and that can be represented as data in the form of programs. It enables the use of algorithms to manipulate, store, and communicate digital information. A computer scientist studies the theory of computation and the practice of designing software systems. Its fields can be divided into theoretical and practical disciplines. Computational complexity theory is highly abstract, while computer graphics emphasizes real-world applications.",
        "https://en.wikipedia.org/wiki/Computer_science",
    ),
    (
        'Data is measured, collected and reported, and analyzed, whereupon it can be visualized using graphs, images or other analysis tools. Data as a general concept refers to the fact that some existing information or knowledge is represented or coded in some form suitable for better usage or processing. Raw data ("unprocessed data") is a collection of numbers or characters before it has been "cleaned" and corrected by researchers.',
        "https://en.wikipedia.org/wiki/Data",
    ),
    (
        'When the data source has a lower-probability value (i.e., when a low-probability event occurs), the event carries more "information" ("surprisal") than when the source data has a higher-probability value. The amount of information conveyed by each event defined in this way becomes a random variable whose expected value is the information entropy. Generally, entropy refers to disorder or uncertainty, and the definition of entropy used in information theory is directly analogous to the definition used in statistical thermodynamics.',
        "https://en.wikipedia.org/wiki/Entropy_(information_theory)",
    ),
    (
        'Information is the resolution of uncertainty; it is that which answers the question of "what an entity is" and is thus that which specifies the nature of that entity, as well as the essentiality of its properties. Information is associated with data and knowledge, as data is meaningful information and represents the values attributed to parameters, and knowledge signifies understanding of an abstract or concrete concept.',
        "https://en.wikipedia.org/wiki/Information",
    ),
    (
        "Thus a neural network is either a biological neural network, made up of real biological neurons, or an artificial neural network, for solving artificial intelligence (AI) problems. The connections of the biological neuron are modeled as weights. A positive weight reflects an excitatory connection, while negative values mean inhibitory connections.",
        "https://en.wikipedia.org/wiki/Neural_network",
    ),
    (
        "In the simplest case, an optimization problem consists of maximizing or minimizing a real function by systematically choosing input values from within an allowed set and computing the value of the function. The generalization of optimization theory and techniques to other formulations constitutes a large area of applied mathematics.",
        "https://en.wikipedia.org/wiki/Mathematical_optimization",
    ),
]


def main():
    with open("template.md", "r") as f:
        template = Template(f.read())

    for ind, paragraph in enumerate(PARAGRAPHS):
        with open(f"mds/{ind}.md", "w") as f:
            f.write(template.render(text=paragraph[0], source=paragraph[1]))


if __name__ == "__main__":
    main()
