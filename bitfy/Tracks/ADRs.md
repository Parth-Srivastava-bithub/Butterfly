# Architecture Decision Records (ADRs)

## Table of Contents

- [ADR 001](#adr-001): Implementing a Fine-Tuned T5-Small Model for Shell Command Generation [`PROPOSED`]
- [ADR 002](#adr-002): Implementing Structured Filtering for Command Generation (Interim Solution) [`ACCEPTED`]

---

<a id="adr-001"></a>
# ADR 001: Implementing a Fine-Tuned T5-Small Model for Shell Command Generation

![Status: PROPOSED](https://img.shields.io/badge/Status-Proposed-yellow)

## Metadata

- **Date**: 2025-05-06
- **Status**: Proposed (as initial architecture)
- **Related ADRs**: [ADR 002](#adr-002) (interim solution while this is developed)

## Context

We are implementing a text-to-shell command generation system using a fine-tuned T5-Small model. The initial goal is to translate natural language input directly into executable shell commands (e.g., `pip install <package>`, `ls -l`). Prompt engineering methods have proven unreliable, generating unpredictable and occasionally crashing shell commands.

## Decision

Implement the T5-Small model for direct shell command generation.

## Consequences

### Pros:
* Potentially allows for natural language interaction with the terminal, simplifying complex tasks.
* Avoids the limitations and unpredictability of prompt engineering.
* Offers a more intuitive and user-friendly interface.

### Cons:
* Model requires significant training data and careful fine-tuning for high accuracy and robustness.
* Potential for errors in command generation, leading to incorrect actions or security vulnerabilities.
* The model's performance may depend on the complexity and clarity of user input.
* Initial implementation may require iterative refinement and retraining.

## Implementation Details

1. **Data Collection and Preparation:** Gather and curate a dataset of natural language queries paired with corresponding shell commands. This dataset will be crucial for training the T5-Small model. The dataset should include various types of shell commands and their associated natural language descriptions.
2. **Model Fine-tuning:** Fine-tune the T5-Small model on the prepared dataset. Experiment with different training hyperparameters (learning rate, batch size, etc.) and validation strategies.
3. **Command Execution Security:** Implement robust security measures to prevent malicious or unintended command execution:
   * **Command Validation:** Before executing a generated command, validate it to ensure it's a permitted and safe command.
   * **Sandboxing (Future):** Consider sandboxing the execution environment to limit the potential damage from a faulty or malicious command.
   * **User Confirmation (Future):** Implement a mechanism for the user to confirm the generated command before execution (especially for sensitive commands).
4. **Error Handling:** Implement mechanisms to handle errors gracefully:
   * **Command Failure Detection:** Monitor the output of executed commands and provide clear error messages to the user if a command fails.
   * **Fallback Mechanism:** If command generation fails or produces an unsafe command, provide a mechanism for the user to rephrase their request, or revert to a structured query approach (described in [ADR 002](#adr-002)).
5. **Evaluation and Iteration:** Continuously evaluate the model's performance. Track accuracy, error rates, and user feedback. Retrain the model regularly with new data and refined parameters.

## Alternatives Considered

* **Prompt Engineering:** Rejected due to its unreliability and unpredictability.
* **Rule-Based Systems:** Considered, but deemed less flexible and scalable than a fine-tuned model.

## Rationale

The fine-tuned T5-Small model offers the best balance between flexibility, natural language understanding, and potential for automation. The decision to use a Transformer model is based on the success of this architecture in similar natural language processing tasks and its capacity to learn the complex mapping between natural language and shell commands.

---

<a id="adr-002"></a>
# ADR 002: Implementing Structured Filtering for Command Generation (Interim Solution)

![Status: ACCEPTED](https://img.shields.io/badge/Status-Proposed-yellow)

## Metadata

- **Date**: 2025-05-06
- **Status**: Accepted (interim solution, deployed before [ADR 001](#adr-001))
- **Related ADRs**: [ADR 001](#adr-001) (long-term solution that will replace this)

## Context

While [ADR 001](#adr-001) focuses on the T5-Small model, its development and deployment will take time. To allow the project to function and handle user requests, we need an interim solution that allows the user to interact with the system and generate commands, even before the fine-tuned model is fully functional. This interim solution must be reliable and provide a user-friendly experience.

## Decision

Implement a structured filtering method for command generation. This method will allow the user to specify parameters and instructions in a structured format, which can then be easily converted into shell commands.

## Consequences

### Pros:
* Provides a reliable and predictable way to generate shell commands.
* Allows the project to function from the beginning, even before the fine-tuned model is ready.
* Easier to implement and maintain than a full command language parser.
* Provides a well-defined and structured interface for command generation.

### Cons:
* Requires users to learn a specific input format (structured parameter inputs).
* Less natural and flexible than the fine-tuned model.
* May require more typing for complex commands.
* Limited in the types of commands it can support initially.

## Implementation Details

1. **Command Definitions:** Define a set of supported commands and their associated parameters. Examples:
   * `--libstatus <library_name> <version_flag>`
   * `--findFile <filename> <search_path>`
   * `--pip install <package_name>`
2. **User Input Format:** Instruct users to input commands using the defined structure. Provide clear examples and documentation.
3. **Input Parsing:** Develop a parsing engine to parse user input and extract parameters. Validate input according to the defined command structure.
4. **Command Generation:** Generate the appropriate shell command based on the parsed parameters.
5. **Command Execution (as in [ADR 001](#adr-001)):** Utilize the same security measures as defined in [ADR 001](#adr-001).
6. **Error Handling:** If the user's input doesn't conform to the defined structure, provide helpful error messages and guidance.
7. **Command Expansion:** As the fine-tuned model is developed, its results can be integrated by automatically expanding from parameters into a generated natural language description. This description can then be used by the prompt model.

## Alternatives Considered

* **Building a Full Command Language Parser:** Rejected because it is significantly more complex and requires more time and resources.
* **Prompt Engineering (with fixed parameters):** Considered, but the structured filtering approach offers more control and predictability.

## Rationale

This structured filtering approach offers a pragmatic solution for handling user input and generating shell commands while the fine-tuned model is being developed. This allows the project to move forward and maintain functionality. The approach offers several advantages, including simplicity of implementation, robustness, and the ability to define clearly defined command structures. This method is the simplest and most predictable and will be useful in creating test cases and edge cases that can be used for the T5 small model.
