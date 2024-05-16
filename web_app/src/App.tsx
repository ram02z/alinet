import "@mantine/core/styles.css";
import {
  ActionIcon,
  Box,
  Collapse,
  Container,
  Flex,
  Group,
  MantineProvider,
  Slider,
  Space,
  Stack,
  Text,
} from "@mantine/core";
import { theme } from "./theme";
import { DragDrop } from "./components/DragDrop";
import { FileList } from "./components/FileList";
import { useState } from "react";
import { Button } from "@mantine/core";
import {
  IconChevronDown,
  IconChevronUp,
  IconSettingsCog,
} from "@tabler/icons-react";
import { v4 as uuidv4 } from "uuid";

import "./App.css";
import { API_URL } from "./env.ts";
import { useDisclosure } from "@mantine/hooks";
import { GeneratedQuestions } from "./components/GeneratedQuestions.tsx";

export interface Question {
  id: string;
  text: string;
  score: number;
  refs: Reference[];
}

export interface Reference {
  file_name: string;
  text: string;
}

export interface FileWithId {
  id: string;
  file: File;
}

export default function App() {
  const [files, setFiles] = useState<FileWithId[]>([]);
  const [selection, setSelection] = useState<string[]>([]);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [topK, setTopK] = useState<number>(1);
  const [distanceThreshold, setDistanceThreshold] = useState<number>(0.5);
  const [openedSettings, { toggle: toggleSettings }] = useDisclosure(false);

  const generateQuestions = async () => {
    setLoading(true);
    const formData = new FormData();
    files.forEach((filesWithId) => {
      formData.append("files", filesWithId.file);
    });

    formData.append("top_k", topK.toString());
    formData.append("distance_threshold", distanceThreshold.toString());

    try {
      const response = await fetch(`${API_URL}/generate_questions`, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        const questionsWithId = data.questions.map(
          (question: {
            text: string;
            similarity_score: number;
            refs: Reference[];
          }) => {
            return {
              id: uuidv4(),
              text: question.text,
              score: question.similarity_score,
              refs: question.refs,
            };
          },
        );
        setQuestions(questionsWithId);
      }
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <MantineProvider theme={theme}>
      <Container fluid p="md">
        <Flex direction="row">
          <DragDrop setFiles={setFiles} />
          <FileList files={files} setFiles={setFiles} />
        </Flex>

        <Box my="md">
          <Group onClick={toggleSettings} style={{ cursor: "pointer" }}>
            <ActionIcon variant="light" color="dark">
              {openedSettings ? <IconChevronUp /> : <IconChevronDown />}
            </ActionIcon>
            <Text tt="uppercase" fw="bold" c="dark" size="lg">
              Configure model generation settings
            </Text>
          </Group>
          <Space h={8} />
          <Collapse in={openedSettings} transitionDuration={0}>
            <Space h={16} />
            <Text size="md" fw="bold">
              Retrieval Augmented Generation (RAG)
            </Text>
            <Space h={8} />
            <Text size="sm">Top K</Text>
            <Slider
              value={topK}
              onChange={setTopK}
              min={1}
              max={16}
              step={1}
              color="cyan"
            />
            <Text size="sm">Similarity Threshold</Text>
            <Slider
              value={distanceThreshold}
              onChange={setDistanceThreshold}
              min={0}
              max={1}
              step={0.1}
              color="cyan"
            />
          </Collapse>
        </Box>

        <Stack justify="center" align="center" mt="xl">
          <Button
            loading={loading}
            onClick={generateQuestions}
            disabled={files.length === 0 || loading}
            leftSection={<IconSettingsCog />}
            size="lg"
            variant="gradient"
            gradient={{ from: "blue", to: "cyan", deg: 90 }}
          >
            Generate Questions
          </Button>
          <Space h={8} />
        </Stack>
        <Text
          size="xl"
          fw={900}
          variant="gradient"
          gradient={{ from: "blue", to: "cyan", deg: 90 }}
        >
          Generated Questions
        </Text>
        <Space h={16} />
        <GeneratedQuestions
          selection={selection}
          setSelection={setSelection}
          questions={questions}
        />
      </Container>
    </MantineProvider>
  );
}
